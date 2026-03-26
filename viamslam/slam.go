// Package viamslam implements a Viam SLAM service that builds a point cloud map
// by accumulating scans from a camera and aligning them with ICP.
package viamslam

import (
	"bytes"
	"context"
	"io"
	"math"
	"sync"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/components/movementsensor"
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

	camera         camera.Camera
	movementSensor movementsensor.MovementSensor
	fsService      framesystem.Service
	icpConfig      icp.ICPConfig
	logger         logging.Logger

	originOnce sync.Once
	originLat  float64
	originLng  float64
	originAlt  float64
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

	var ms movementsensor.MovementSensor
	if cfg.MovementSensor != "" {
		ms, err = movementsensor.FromDependencies(deps, cfg.MovementSensor)
		if err != nil {
			return nil, err
		}
	}

	return &ICPSlam{
		Named:          conf.ResourceName().AsNamed(),
		camera:         cam,
		movementSensor: ms,
		fsService:      fs,
		icpConfig:      icp.DefaultICPConfig(),
		logger:         logger,
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

	// Transform the scan from camera frame into the frame system's root (base frame).
	cameraFrame := s.camera.Name().ShortName()
	basePC, err := s.fsService.TransformPointCloud(ctx, pc, cameraFrame, referenceframe.World)
	if err != nil {
		return nil, err
	}

	// If a movement sensor is configured, apply its pose to place the scan
	// in the global frame. Without this, every scan lands at the base origin.
	worldPC := basePC
	if s.movementSensor != nil {
		worldPC, err = s.applyMovementSensorPose(ctx, basePC)
		if err != nil {
			return nil, err
		}
	}

	// PCD files use meters; ICP works in millimeters. Convert here on ingestion.
	mmPC, err := scalePointCloud(worldPC, metersToMM)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.clouds = append(s.clouds, mmPC)
	numScans := len(s.clouds)

	// Merge the new scan into the existing map (no ICP alignment).
	if s.mergedPC == nil {
		s.mergedPC = mmPC
	} else {
		merged, err := icp.NaiveMergeClouds(ctx, s.mergedPC, mmPC, s.logger)
		if err != nil {
			s.mu.Unlock()
			return nil, err
		}
		s.mergedPC = merged
	}
	size := s.mergedPC.Size()
	s.mu.Unlock()

	return map[string]interface{}{
		"scans_accumulated": numScans,
		"map_points":        size,
	}, nil
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

// applyMovementSensorPose gets the current position and orientation from the movement
// sensor and transforms the point cloud from the base frame into the global frame.
// The first call records the origin; subsequent calls compute a local offset from it.
func (s *ICPSlam) applyMovementSensorPose(ctx context.Context, pc pointcloud.PointCloud) (pointcloud.PointCloud, error) {
	geoPoint, alt, err := s.movementSensor.Position(ctx, nil)
	if err != nil {
		return nil, err
	}

	orientation, err := s.movementSensor.Orientation(ctx, nil)
	if err != nil {
		s.logger.Warnf("movement sensor orientation unavailable, using identity: %v", err)
		orientation = spatialmath.NewZeroPose().Orientation()
	}

	// Record the first position as origin.
	s.originOnce.Do(func() {
		s.originLat = geoPoint.Lat()
		s.originLng = geoPoint.Lng()
		s.originAlt = alt
	})

	// Convert geo coordinates to local meter offsets from origin.
	latRad := s.originLat * math.Pi / 180.0
	dx := (geoPoint.Lng() - s.originLng) * math.Cos(latRad) * 111319.5
	dy := (geoPoint.Lat() - s.originLat) * 111319.5
	dz := alt - s.originAlt

	pose := spatialmath.NewPose(r3.Vector{X: dx, Y: dy, Z: dz}, orientation)
	return transformPointCloud(pc, pose)
}

// transformPointCloud applies a pose transform to every point in the cloud.
func transformPointCloud(src pointcloud.PointCloud, pose spatialmath.Pose) (pointcloud.PointCloud, error) {
	dst := pointcloud.NewBasicEmpty()
	var iterErr error
	src.Iterate(0, 0, func(p r3.Vector, d pointcloud.Data) bool {
		transformed := spatialmath.Compose(pose, spatialmath.NewPoseFromPoint(p)).Point()
		if err := dst.Set(transformed, d); err != nil {
			iterErr = err
			return false
		}
		return true
	})
	return dst, iterErr
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
