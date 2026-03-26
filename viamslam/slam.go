// Package viamslam implements a Viam SLAM service that builds a point cloud map
// by accumulating scans from a camera and aligning them with ICP.
package viamslam

import (
	"bytes"
	"context"
	"io"
	"sync"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/robot/framesystem"
	"go.viam.com/rdk/services/slam"
	"go.viam.com/rdk/spatialmath"

	icp "github.com/viamrobotics/icp"
)

const (
	chunkSizeBytes = 1 * 1024 * 1024 // 1MB
	metersToMM     = 1000.0
	mmToMeters     = 1.0 / metersToMM

	// DoCommand key for adding a new scan to the map.
	addScanKey = "add_scan"
)

// Model is the resource model triplet for this SLAM service.
var Model = resource.NewModel("cjnj193", "icp", "icp-slam")

func init() {
	resource.RegisterService(
		slam.API,
		Model,
		resource.Registration[slam.Service, *Config]{
			Constructor: newICPSlam,
		},
	)
}

// Config holds the names of the camera and optional movement sensor to use.
type Config struct {
	Camera         string `json:"camera"`
	MovementSensor string `json:"movement_sensor,omitempty"`
}

// Validate returns the list of dependencies this service requires.
func (c *Config) Validate(path string) ([]string, []string, error) {
	if c.Camera == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "camera")
	}
	deps := []string{
		c.Camera,
		framesystem.PublicServiceName.String(),
	}
	if c.MovementSensor != "" {
		deps = append(deps, c.MovementSensor)
	}
	return deps, nil, nil
}

// ICPSlam is a SLAM service that accumulates scans and aligns them with ICP.
type ICPSlam struct {
	resource.Named
	resource.AlwaysRebuild

	mu       sync.RWMutex
	clouds   []pointcloud.PointCloud // accumulated world-frame scans
	mergedPC pointcloud.PointCloud   // latest ICP-aligned map

	camera    camera.Camera
	fsService framesystem.Service
	icpConfig icp.ICPConfig
	logger    logging.Logger
}

func newICPSlam(
	_ context.Context,
	deps resource.Dependencies,
	conf resource.Config,
	logger logging.Logger,
) (slam.Service, error) {
	cfg, err := resource.NativeConfig[*Config](conf)
	if err != nil {
		return nil, err
	}

	cam, err := camera.FromDependencies(deps, cfg.Camera)
	if err != nil {
		return nil, err
	}

	fs, err := framesystem.FromDependencies(deps)
	if err != nil {
		return nil, err
	}

	return &ICPSlam{
		Named:     conf.ResourceName().AsNamed(),
		camera:    cam,
		fsService: fs,
		icpConfig: icp.DefaultICPConfig(),
		logger:    logger,
	}, nil
}

// DoCommand handles the "add_scan" command, which pulls the current point cloud
// from the camera, transforms it into the world frame via the frame system,
// adds it to the accumulated scans, and re-runs ICP alignment.
func (s *ICPSlam) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	if _, ok := cmd[addScanKey]; !ok {
		return map[string]interface{}{}, nil
	}

	// Pull the current scan from the camera (in camera frame).
	pc, err := s.camera.NextPointCloud(ctx)
	if err != nil {
		return nil, err
	}

	// Transform the scan from camera frame into world frame using the frame system.
	// The frame system uses the robot's configured transforms (including any odometry
	// supplemental transforms if the robot is set up with a movement sensor).
	cameraFrame := s.camera.Name().ShortName()
	worldPC, err := s.fsService.TransformPointCloud(ctx, pc, cameraFrame, referenceframe.World)
	if err != nil {
		return nil, err
	}

	// PCD files use meters; ICP works in millimeters. Convert here on ingestion.
	mmPC, err := scalePointCloud(worldPC, metersToMM)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.clouds = append(s.clouds, mmPC)
	// Always keep the latest scan visible in PointCloudMap, even before ICP runs.
	if s.mergedPC == nil {
		s.mergedPC = mmPC
	}
	clouds := make([]pointcloud.PointCloud, len(s.clouds))
	copy(clouds, s.clouds)
	s.mu.Unlock()

	s.logger.Infof("add_scan: %d scans accumulated (%d points in latest scan)", len(clouds), mmPC.Size())

	resp := map[string]interface{}{
		"scans_accumulated": len(clouds),
	}

	// Run ICP alignment once we have at least two scans.
	if len(clouds) >= 2 {
		s.logger.Infof("add_scan: running ICP alignment across %d scans", len(clouds))
		merged, err := icp.AlignPointClouds(ctx, clouds, s.icpConfig, s.logger)
		if err != nil {
			// Log the failure but don't return it — the scan is already accumulated
			// and ICP will retry on the next add_scan call.
			s.logger.Errorw("add_scan: ICP alignment failed", "error", err, "num_scans", len(clouds))
			resp["icp_error"] = err.Error()
		} else {
			s.mu.Lock()
			s.mergedPC = merged
			s.mu.Unlock()
			s.logger.Infof("add_scan: ICP succeeded, merged map has %d points", merged.Size())
		}
	}

	s.mu.RLock()
	resp["map_points"] = s.mergedPC.Size()
	s.mu.RUnlock()

	return resp, nil
}

// PointCloudMap serializes the current merged map to PCD binary format and returns
// it as a streaming callback that yields 1MB chunks until io.EOF.
func (s *ICPSlam) PointCloudMap(_ context.Context, _ bool) (func() ([]byte, error), error) {
	s.mu.RLock()
	cloud := s.mergedPC
	s.mu.RUnlock()

	var buf bytes.Buffer
	if cloud != nil {
		// ICP map is in mm; convert back to meters for PCD output.
		metersCloud, err := scalePointCloud(cloud, mmToMeters)
		if err != nil {
			return nil, err
		}
		if err := pointcloud.ToPCD(metersCloud, &buf, pointcloud.PCDBinary); err != nil {
			return nil, err
		}
	}

	data := buf.Bytes()
	reader := bytes.NewReader(data)
	chunk := make([]byte, chunkSizeBytes)

	return func() ([]byte, error) {
		n, err := reader.Read(chunk)
		return chunk[:n], err
	}, nil
}

// Position returns a zero pose — localization is not implemented.
func (s *ICPSlam) Position(_ context.Context) (spatialmath.Pose, error) {
	return spatialmath.NewZeroPose(), nil
}

// InternalState returns an empty stream.
func (s *ICPSlam) InternalState(_ context.Context) (func() ([]byte, error), error) {
	return func() ([]byte, error) {
		return nil, io.EOF
	}, nil
}

// Properties describes the capabilities of this SLAM service.
func (s *ICPSlam) Properties(_ context.Context) (slam.Properties, error) {
	return slam.Properties{
		CloudSlam:             false,
		MappingMode:           slam.MappingModeNewMap,
		InternalStateFileType: "",
		SensorInfo: []slam.SensorInfo{
			{Name: s.camera.Name().ShortName(), Type: slam.SensorTypeCamera},
		},
	}, nil
}

// Close is a no-op.
func (s *ICPSlam) Close(_ context.Context) error {
	return nil
}

// scalePointCloud returns a new point cloud with every coordinate multiplied by factor.
func scalePointCloud(src pointcloud.PointCloud, factor float64) (pointcloud.PointCloud, error) {
	dst := pointcloud.NewBasicEmpty()
	var iterErr error
	src.Iterate(0, 0, func(p r3.Vector, d pointcloud.Data) bool {
		scaled := r3.Vector{X: p.X * factor, Y: p.Y * factor, Z: p.Z * factor}
		if err := dst.Set(scaled, d); err != nil {
			iterErr = err
			return false
		}
		return true
	})
	return dst, iterErr
}
