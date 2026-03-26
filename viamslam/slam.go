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
	"go.viam.com/rdk/components/movementsensor"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/slam"
	"go.viam.com/rdk/spatialmath"

	icp "github.com/viamrobotics/icp"
)

const (
	chunkSizeBytes = 1 * 1024 * 1024 // 1MB

	// DoCommand key for adding a new scan to the map.
	addScanKey = "add_scan"

	// Key used with wheeled odometry's Position() to get local XY in meters.
	returnRelativePosKey = "return_relative_pos_m"
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
	UseICP         *bool  `json:"use_icp,omitempty"`
}

// Validate returns the list of dependencies this service requires.
func (c *Config) Validate(path string) ([]string, []string, error) {
	if c.Camera == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "camera")
	}
	deps := []string{c.Camera}
	if c.MovementSensor != "" {
		deps = append(deps, c.MovementSensor)
	}
	return deps, nil, nil
}

// ICPSlam is a SLAM service that accumulates scans and aligns them with ICP.
type ICPSlam struct {
	resource.Named
	resource.AlwaysRebuild

	mu        sync.RWMutex
	clouds    []pointcloud.PointCloud // accumulated world-frame scans
	mergedPC  pointcloud.PointCloud   // latest merged map
	lastPose  spatialmath.Pose        // last pose recorded from the movement sensor

	camera         camera.Camera
	movementSensor movementsensor.MovementSensor // optional; nil if not configured
	icpConfig      icp.ICPConfig
	useICP         bool
	logger         logging.Logger
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

	var ms movementsensor.MovementSensor
	if cfg.MovementSensor != "" {
		ms, err = movementsensor.FromDependencies(deps, cfg.MovementSensor)
		if err != nil {
			return nil, err
		}
	}

	useICP := true
	if cfg.UseICP != nil {
		useICP = *cfg.UseICP
	}

	return &ICPSlam{
		Named:          conf.ResourceName().AsNamed(),
		camera:         cam,
		movementSensor: ms,
		icpConfig:      icp.DefaultICPConfig(),
		useICP:         useICP,
		logger:         logger,
	}, nil
}

// DoCommand handles the "add_scan" command, which pulls the current point cloud
// from the camera, optionally transforms it into the world frame via the movement
// sensor pose, adds it to the accumulated scans, and re-runs alignment.
func (s *ICPSlam) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	if _, ok := cmd[addScanKey]; !ok {
		return map[string]interface{}{}, nil
	}

	// Pull the current scan from the camera (in camera frame, meters).
	pc, err := s.camera.NextPointCloud(ctx)
	if err != nil {
		return nil, err
	}

	// If a movement sensor is configured, transform the scan into world frame
	// using the sensor's current pose.
	worldPC := pc
	if s.movementSensor != nil {
		pose, err := sensorPose(ctx, s.movementSensor)
		if err != nil {
			return nil, err
		}
		dst := pointcloud.NewBasicEmpty()
		if err := pointcloud.ApplyOffset(pc, pose, dst); err != nil {
			return nil, err
		}
		worldPC = dst
		ov := pose.Orientation().OrientationVectorDegrees()
		pt := pose.Point()
		s.logger.Infof("add_scan: sensor pose — x: %.4f y: %.4f z: %.4f ox: %.4f oy: %.4f oz: %.4f th: %.4f",
			pt.X, pt.Y, pt.Z, ov.OX, ov.OY, ov.OZ, ov.Theta)
		s.mu.Lock()
		s.lastPose = pose
		s.mu.Unlock()
	}

	s.mu.Lock()
	s.clouds = append(s.clouds, worldPC)
	// Always keep the latest scan visible in PointCloudMap, even before alignment runs.
	if s.mergedPC == nil {
		s.mergedPC = worldPC
	}
	clouds := make([]pointcloud.PointCloud, len(s.clouds))
	copy(clouds, s.clouds)
	s.mu.Unlock()

	s.logger.Infof("add_scan: %d scans accumulated (%d points in latest scan)", len(clouds), worldPC.Size())

	resp := map[string]interface{}{
		"scans_accumulated": len(clouds),
	}

	// Merge or align scans once we have at least two.
	if len(clouds) >= 2 {
		var merged pointcloud.PointCloud
		var mergeErr error
		if s.useICP {
			s.logger.Infof("add_scan: running ICP alignment across %d scans", len(clouds))
			merged, mergeErr = icp.AlignPointClouds(ctx, clouds, s.icpConfig, s.logger)
		} else {
			s.logger.Infof("add_scan: naive merge across %d scans (ICP disabled)", len(clouds))
			merged = clouds[0]
			for _, c := range clouds[1:] {
				merged, mergeErr = icp.NaiveMergeClouds(ctx, merged, c, s.logger)
				if mergeErr != nil {
					break
				}
			}
		}
		if mergeErr != nil {
			s.logger.Errorw("add_scan: merge failed", "error", mergeErr, "num_scans", len(clouds))
			resp["merge_error"] = mergeErr.Error()
		} else {
			s.mu.Lock()
			s.mergedPC = merged
			s.mu.Unlock()
			s.logger.Infof("add_scan: merge succeeded, map has %d points", merged.Size())
		}
	}

	s.mu.RLock()
	if s.mergedPC != nil {
		resp["map_points"] = s.mergedPC.Size()
	}
	s.mu.RUnlock()

	return resp, nil
}

// sensorPose reads the current position and orientation from a movement sensor
// and returns them as a spatialmath.Pose. Position is read in local meters
// (return_relative_pos_m) so the pose is relative to the sensor's origin.
func sensorPose(ctx context.Context, ms movementsensor.MovementSensor) (spatialmath.Pose, error) {
	pos, alt, err := ms.Position(ctx, map[string]interface{}{returnRelativePosKey: true})
	if err != nil {
		return nil, err
	}
	orient, err := ms.Orientation(ctx, nil)
	if err != nil {
		return nil, err
	}
	// Wheeled odometry returns local XY as Lat=Y, Lng=X (in millimeters).
	translation := r3.Vector{X: pos.Lng(), Y: pos.Lat(), Z: alt}
	return spatialmath.NewPose(translation, orient), nil
}

// PointCloudMap serializes the current merged map to PCD binary format and returns
// it as a streaming callback that yields 1MB chunks until io.EOF.
func (s *ICPSlam) PointCloudMap(_ context.Context, _ bool) (func() ([]byte, error), error) {
	s.mu.RLock()
	cloud := s.mergedPC
	s.mu.RUnlock()

	if cloud == nil {
		cloud = pointcloud.NewBasicEmpty()
	}

	// pointcloud.ToPCD handles the mm→meters conversion internally.
	var buf bytes.Buffer
	if err := pointcloud.ToPCD(cloud, &buf, pointcloud.PCDBinary); err != nil {
		return nil, err
	}

	data := buf.Bytes()
	reader := bytes.NewReader(data)
	chunk := make([]byte, chunkSizeBytes)

	return func() ([]byte, error) {
		n, err := reader.Read(chunk)
		return chunk[:n], err
	}, nil
}

// Position returns the last pose recorded from the movement sensor, or a zero
// pose if no scan has been added yet.
func (s *ICPSlam) Position(_ context.Context) (spatialmath.Pose, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.lastPose == nil {
		return spatialmath.NewZeroPose(), nil
	}
	return s.lastPose, nil
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
