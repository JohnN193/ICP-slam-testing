// Package viamslam implements a Viam SLAM service that exposes an ICP-merged point cloud map.
package viamslam

import (
	"bytes"
	"context"
	"io"
	"sync"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/slam"
	"go.viam.com/rdk/spatialmath"
)

const chunkSizeBytes = 1 * 1024 * 1024 // 1MB

// Model is the resource model triplet for this SLAM service.
var Model = resource.NewModel("viam-labs", "icp", "icp-slam")

func init() {
	resource.RegisterService(
		slam.API,
		Model,
		resource.Registration[slam.Service, resource.NoNativeConfig]{
			Constructor: newICPSlam,
		},
	)
}

// ICPSlam is a SLAM service that serves a pre-built ICP point cloud map.
type ICPSlam struct {
	resource.Named
	resource.AlwaysRebuild

	mu     sync.RWMutex
	cloud  pointcloud.PointCloud
	logger logging.Logger
}

func newICPSlam(
	_ context.Context,
	_ resource.Dependencies,
	conf resource.Config,
	logger logging.Logger,
) (slam.Service, error) {
	return &ICPSlam{
		Named:  conf.ResourceName().AsNamed(),
		logger: logger,
	}, nil
}

// SetPointCloud updates the point cloud map stored in this service.
func (s *ICPSlam) SetPointCloud(cloud pointcloud.PointCloud) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cloud = cloud
}

// Position returns a zero pose — no localization is implemented.
func (s *ICPSlam) Position(_ context.Context) (spatialmath.Pose, error) {
	return spatialmath.NewZeroPose(), nil
}

// PointCloudMap serializes the stored point cloud to PCD binary format and returns it as
// a streaming callback that yields 1MB chunks until io.EOF.
func (s *ICPSlam) PointCloudMap(_ context.Context, _ bool) (func() ([]byte, error), error) {
	s.mu.RLock()
	cloud := s.cloud
	s.mu.RUnlock()

	var buf bytes.Buffer
	if cloud != nil {
		if err := pointcloud.ToPCD(cloud, &buf, pointcloud.PCDBinary); err != nil {
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

// InternalState returns an empty stream — no internal algorithm state to expose.
func (s *ICPSlam) InternalState(_ context.Context) (func() ([]byte, error), error) {
	return func() ([]byte, error) {
		return nil, io.EOF
	}, nil
}

// Properties returns the properties of this SLAM service.
func (s *ICPSlam) Properties(_ context.Context) (slam.Properties, error) {
	return slam.Properties{
		CloudSlam:             false,
		MappingMode:           slam.MappingModeLocalizationOnly,
		InternalStateFileType: "",
		SensorInfo:            []slam.SensorInfo{},
	}, nil
}

// DoCommand is a no-op for now.
func (s *ICPSlam) DoCommand(_ context.Context, _ map[string]interface{}) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}

// Close is a no-op.
func (s *ICPSlam) Close(_ context.Context) error {
	return nil
}
