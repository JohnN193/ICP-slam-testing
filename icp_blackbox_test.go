package icp_test

// This package is for tests that do not rely on access to the internal implementation of the icp
// package. Ideally, all icp tests will eventually be moved in here, but many are currently in
// icp_test.go instead.

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"testing"

	"github.com/golang/geo/r3"
	vizClient "github.com/viam-labs/motion-tools/client/client"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/test"

	"github.com/viamrobotics/icp"
	"github.com/viamrobotics/icp/pcutils"
)

func init() {
	// pointcloud merging uses a fair bit of memory, we need to not crash the github action runner
	debug.SetGCPercent(40)
}

func TestICPAlignmentMultiCloud(t *testing.T) {
	ctx := context.Background()

	// Create deterministic random source for reproducible tests
	pointCount := 3000
	cloudCount := 5
	rng := rand.New(rand.NewSource(42)) //nolint:gosec

	// Create 5 overlapping point clouds from the same surface with different misalignments
	clouds := make([]pointcloud.PointCloud, cloudCount)
	thetaStart, thetaEnd := 0.0, math.Pi // 0-180 degrees
	zStart, zEnd := 0.0, 100.0           // 0-100mm height

	// Cloud 0: Reference (no misalignment)
	basePoints := createCameraViewOfSurface(thetaStart, thetaEnd, zStart, zEnd, pointCount, rng)
	clouds[0] = pointcloud.NewBasicEmpty()
	for _, p := range basePoints {
		err := clouds[0].Set(p, pointcloud.NewBasicData())
		test.That(t, err, test.ShouldBeNil)
	}

	// Clouds 1-4: Apply different misalignments
	misalignments := []spatialmath.Pose{
		spatialmath.NewPose(r3.Vector{X: 1.5, Y: -0.8, Z: 0.3}, &spatialmath.OrientationVectorDegrees{OX: 4.6, OY: -2.9, OZ: 6.9}),
		spatialmath.NewPose(r3.Vector{X: -1.2, Y: 0.6, Z: -0.4}, &spatialmath.OrientationVectorDegrees{OX: -3.4, OY: 4.0, OZ: -5.2}),
		spatialmath.NewPose(r3.Vector{X: 0.9, Y: 1.1, Z: 0.2}, &spatialmath.OrientationVectorDegrees{OX: 2.3, OY: -4.6, OZ: 3.4}),
		spatialmath.NewPose(r3.Vector{X: -0.7, Y: -0.5, Z: 0.8}, &spatialmath.OrientationVectorDegrees{OX: -1.7, OY: 2.9, OZ: -6.3}),
	}

	for i := 1; i < cloudCount; i++ {
		basePoints := createCameraViewOfSurface(thetaStart, thetaEnd, zStart, zEnd, pointCount, rng)
		clouds[i] = pointcloud.NewBasicEmpty()

		for _, p := range basePoints {
			// Apply specific misalignment
			pointPose := spatialmath.NewPoseFromPoint(p)
			transformedPose := spatialmath.Compose(misalignments[i-1], pointPose)
			transformedPoint := transformedPose.Point()
			err := clouds[i].Set(transformedPoint, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}
	}

	// Visualize all original clouds
	colors := [][3]uint8{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}}
	for i := 0; i < 5; i++ {
		cloudName := fmt.Sprintf("multicloud_%d_original", i)
		if err := vizClient.DrawPointCloud(cloudName, clouds[i], &colors[i]); err != nil {
			t.Logf("Warning: Failed to visualize cloud%d: %v", i, err)
		}
	}

	// Run multi-cloud ICP alignment
	config := icp.DefaultICPConfig()
	config.MinOverlapPoints = 35
	logger := logging.NewTestLogger(t)

	mergedCloud, err := icp.AlignPointClouds(ctx, clouds, config, logger)
	test.That(t, err, test.ShouldBeNil)

	// Verify results
	test.That(t, mergedCloud, test.ShouldNotBeNil)

	// Visualize merged result
	if err := vizClient.DrawPointCloud("multicloud_merged_result", mergedCloud, nil); err != nil {
		t.Logf("Warning: Failed to visualize: %v", err)
	}

	// Check that merged cloud has correct number of points
	expectedMinPoints := cloudCount * pointCount // Should have points from all 5 clouds
	test.That(t, mergedCloud.Size(), test.ShouldEqual, expectedMinPoints)
}

func TestICPAlignmentReal(t *testing.T) {
	ctx := context.Background()

	// Test configuration
	config := icp.DefaultICPConfig()
	config.FinalRealign = false
	logger := logging.NewTestLogger(t)

	// Load and visualize original point clouds before alignment
	filePaths := []string{}
	// NOTE: If you wish, any pointclouds from any real run could be put in the test folder, and then the next two lines updated
	// to specify them properly if you wish to rerun this algorithm locally.
	for i := 0; i <= 2; i++ {
		filePaths = append(filePaths, "blue_wrld"+strconv.Itoa(i)+".pcd")
	}
	fullPaths := []string{}

	clouds := make([]pointcloud.PointCloud, len(filePaths))
	colors := [][3]uint8{
		{255, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
		{255, 0, 255},
		{0, 255, 255},
		{255, 0, 0},
		{0, 127, 0},
		{0, 0, 127},
		{127, 0, 127},
		{0, 127, 127},
		{127, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
		{255, 0, 255},
		{0, 255, 255},
	}

	for i, filePathRaw := range filePaths {
		_, filename, _, _ := runtime.Caller(0)
		projectRoot := filepath.Dir(filename)
		pcdFile := filepath.Join(projectRoot, "test", filePathRaw)
		fullPaths = append(fullPaths, pcdFile)
		cloud, err := pointcloud.NewFromFile(pcdFile, "")
		test.That(t, err, test.ShouldBeNil)
		clouds[i] = cloud

		cloudName := fmt.Sprintf("cloud%d_real_original", i)
		if err := vizClient.DrawPointCloud(cloudName, cloud, &colors[i]); err != nil {
			t.Logf("Warning: Failed to visualize %s: %v", cloudName, err)
		}
	}

	// Run ICP alignment
	mergedCloud, err := icp.AlignPointCloudsFromFiles(ctx, fullPaths, config, logger)
	test.That(t, err, test.ShouldBeNil)

	// Verify results
	test.That(t, mergedCloud, test.ShouldNotBeNil)
}

// computeMedianZ is a helper function which calculates the median Z coordinate of a set of points.
// Returns a point where X and Y are the centroid values but Z is the median.
func computeMedianZ(points []r3.Vector) r3.Vector {
	if len(points) == 0 {
		return r3.Vector{}
	}

	// Calculate centroid for X and Y coordinates
	centroid := pcutils.ComputeCentroid(points)

	// Extract and sort Z values
	zValues := make([]float64, len(points))
	for i, p := range points {
		zValues[i] = p.Z
	}
	sort.Float64s(zValues)

	// Calculate median Z
	var medianZ float64
	n := len(zValues)
	if n%2 == 0 {
		// Even number of points: average of two middle values
		medianZ = (zValues[n/2-1] + zValues[n/2]) / 2.0
	} else {
		// Odd number of points: middle value
		medianZ = zValues[n/2]
	}

	return r3.Vector{X: centroid.X, Y: centroid.Y, Z: medianZ}
}

// tests.
// TODO(RSDK-11818): Change TestICPAlignmentFlat test to use evaluation.EvaluateFlatness
// Blocked by import cycles: RSDK-11817.
func TestICPAlignmentFlat(t *testing.T) {
	t.Skip() // Takes too long to run as part of CI
	// Test configuration

	ctx := context.Background()

	config := icp.DefaultICPConfig()
	centroidDist := 80.
	logger := logging.NewTestLogger(t)

	// Load and visualize original point clouds before alignment
	filePaths := []string{}
	// NOTE: If you wish, any pointclouds from any real run could be put in the test folder, and then the next two lines updated
	// to specify them properly if you wish to rerun this algorithm locally.
	for i := 0; i <= 4; i++ {
		filePaths = append(filePaths, "flat_world_"+strconv.Itoa(i)+".pcd")
	}

	clouds := make([]pointcloud.PointCloud, len(filePaths))
	colors := [][3]uint8{
		{255, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
		{255, 0, 255},
		{0, 255, 255},
		{255, 0, 0},
		{0, 127, 0},
		{0, 0, 127},
		{127, 0, 127},
		{0, 127, 127},
		{127, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
		{255, 0, 255},
		{0, 255, 255},
	}

	for i, filePathRaw := range filePaths {
		_, filename, _, _ := runtime.Caller(0)
		projectRoot := filepath.Dir(filename)
		pcdFile := filepath.Join(projectRoot, "test", filePathRaw)
		cloud, err := pointcloud.NewFromFile(pcdFile, "")
		test.That(t, err, test.ShouldBeNil)
		clouds[i] = cloud

		cloudName := fmt.Sprintf("cloud%d_real_original", i)
		if err := vizClient.DrawPointCloud(cloudName, cloud, &colors[i]); err != nil {
			t.Logf("Warning: Failed to visualize %s: %v", cloudName, err)
		}
	}

	// Create naive merge of all clouds for comparison
	var naiveMerge pointcloud.PointCloud = clouds[0]
	var err error
	for i := 1; i < len(clouds); i++ {
		naiveMerge, err = icp.NaiveMergeClouds(ctx, naiveMerge, clouds[i], logger)
		test.That(t, err, test.ShouldBeNil)
	}

	naiveOctree, err := pointcloud.ToBasicOctree(naiveMerge, 0)
	test.That(t, err, test.ShouldBeNil)
	allNaivePoints := naiveOctree.ToPoints(0)
	planeCentroid := computeMedianZ(allNaivePoints)

	t.Logf("Using plane point: [%.2f, %.2f, %.2f]", planeCentroid.X, planeCentroid.Y, planeCentroid.Z)

	// Find points within distance of centroid in naive merge and compute plane normal
	nearbyPointsNaive, err := naiveOctree.PointsWithinRadius(planeCentroid, centroidDist)
	test.That(t, err, test.ShouldBeNil)

	var naivePlaneNormal r3.Vector
	if len(nearbyPointsNaive) >= 3 {
		// Compute plane normal using principal directions
		principalDir, secondaryDir, err := pcutils.ComputePrincipalDirections(nearbyPointsNaive, planeCentroid)
		if err == nil {
			naivePlaneNormal = principalDir.Cross(secondaryDir).Normalize()
		} else {
			naivePlaneNormal = r3.Vector{X: 0, Y: 0, Z: 1} // fallback
		}
	} else {
		naivePlaneNormal = r3.Vector{X: 0, Y: 0, Z: 1} // fallback
	}
	t.Logf("Naive merge plane normal: [%.3f, %.3f, %.3f] (from %d nearby points)",
		naivePlaneNormal.X, naivePlaneNormal.Y, naivePlaneNormal.Z, len(nearbyPointsNaive))

	naivePlaneFitness, err := pcutils.MeasurePlaneFitness(naiveOctree, pcutils.ComputeCentroid(nearbyPointsNaive), naivePlaneNormal)
	test.That(t, err, test.ShouldBeNil)
	t.Logf("Naive merge plane fitness (RMS distance): %.3f mm", naivePlaneFitness)

	// Run ICP alignment
	mergedCloud, err := icp.AlignPointClouds(ctx, clouds, config, logger)
	test.That(t, err, test.ShouldBeNil)

	// Verify results
	test.That(t, mergedCloud, test.ShouldNotBeNil)

	// Find points within 10mm of centroid in aligned merge and compute plane normal
	alignedOctree, err := pointcloud.ToBasicOctree(mergedCloud, 0)
	test.That(t, err, test.ShouldBeNil)
	planeCentroid = computeMedianZ(alignedOctree.ToPoints(0))
	nearbyPointsAligned, err := alignedOctree.PointsWithinRadius(planeCentroid, centroidDist)
	test.That(t, err, test.ShouldBeNil)
	t.Logf("Using plane point: [%.2f, %.2f, %.2f]", planeCentroid.X, planeCentroid.Y, planeCentroid.Z)

	var alignedPlaneNormal r3.Vector
	if len(nearbyPointsAligned) >= 3 {
		// Compute plane normal using principal directions
		principalDir, secondaryDir, err := pcutils.ComputePrincipalDirections(nearbyPointsAligned, planeCentroid)
		if err == nil {
			alignedPlaneNormal = principalDir.Cross(secondaryDir).Normalize()
		} else {
			alignedPlaneNormal = r3.Vector{X: 0, Y: 0, Z: 1} // fallback
		}
	} else {
		alignedPlaneNormal = r3.Vector{X: 0, Y: 0, Z: 1} // fallback
	}
	t.Logf("Aligned merge plane normal: [%.3f, %.3f, %.3f] (from %d nearby points)",
		alignedPlaneNormal.X, alignedPlaneNormal.Y, alignedPlaneNormal.Z, len(nearbyPointsAligned))
	alignedPlaneFitness, err := pcutils.MeasurePlaneFitness(alignedOctree, planeCentroid, alignedPlaneNormal)
	test.That(t, err, test.ShouldBeNil)
	t.Logf("Aligned merge plane fitness (RMS distance): %.3f mm", alignedPlaneFitness)

	// Calculate improvement
	improvement := naivePlaneFitness - alignedPlaneFitness
	improvementPercent := (improvement / naivePlaneFitness) * 100
	t.Logf("Plane fitness improvement: %.3f mm (%.1f%% better)", improvement, improvementPercent)

	pcdSlice := pointcloud.CloudToPoints(mergedCloud)
	downsampledCloudFinal := pcutils.VoxelDownsample(pcdSlice, 5)
	overlapOctree, err := pcutils.VectorsToOctree(downsampledCloudFinal)
	test.That(t, err, test.ShouldBeNil)

	// Visualize aligned result
	if err := vizClient.DrawPointCloud("merged_real_result", overlapOctree, nil); err != nil {
		t.Logf("Warning: Failed to visualize real merged cloud: %v", err)
	}
}

// Create a realistic camera view of a curved surface section.
// Note that this is duplicated in icp_test.go.
func createCameraViewOfSurface(thetaStart, thetaEnd, zStart, zEnd float64, numPoints int, rng *rand.Rand) []r3.Vector {
	// Generate surface points
	points := generateSurfaceSection(thetaStart, thetaEnd, zStart, zEnd, numPoints, rng)

	// Add some measurement noise typical of camera data
	for i, p := range points {
		noise := r3.Vector{
			X: (rng.Float64() - 0.5) * 0.5, // ±0.25mm noise
			Y: (rng.Float64() - 0.5) * 0.5,
			Z: (rng.Float64() - 0.5) * 0.5,
		}
		points[i] = p.Add(noise)
	}

	return points
}

// Generate points on a section of a curved surface (cylindrical pipe-like surface).
// Note that this is duplicated in icp_test.go.
func generateSurfaceSection(thetaStart, thetaEnd, zStart, zEnd float64, numPoints int, rng *rand.Rand) []r3.Vector {
	var points []r3.Vector
	baseRadius := 450.0 // 45mm base radius

	for i := 0; i < numPoints; i++ {
		// Random position within the camera's view region
		theta := thetaStart + rng.Float64()*(thetaEnd-thetaStart)
		z := zStart + rng.Float64()*(zEnd-zStart)

		// Add some surface variation to make it more realistic (not perfect cylinder)
		radiusVariation := 2.0 * (0.5 + 0.3*math.Sin(3*theta) + 0.2*math.Cos(5*z/10))
		radius := baseRadius + radiusVariation

		// Convert to Cartesian coordinates
		x := radius * math.Cos(theta)
		y := radius * math.Sin(theta)

		points = append(points, r3.Vector{X: x, Y: y, Z: z})
	}

	return points
}

// This test creates.
func TestICPAlignmentVShape(t *testing.T) {
	ctx := context.Background()

	// Create two perfect flat rectangles that intersect at a slight V angle
	config := icp.DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	// Create two rectangular point clouds forming a V shape
	cloud1, cloud2 := createVShapeRectangles(t)

	// Visualize original clouds
	err := vizClient.DrawPointCloud("v_cloud1_original", cloud1, &[3]uint8{255, 0, 0})
	if err != nil {
		t.Logf("Warning: Failed to visualize: %v", err)
	}
	err = vizClient.DrawPointCloud("v_cloud2_original", cloud2, &[3]uint8{0, 255, 0})
	if err != nil {
		t.Logf("Warning: Failed to visualize: %v", err)
	}

	// Test alignment
	clouds := []pointcloud.PointCloud{cloud1, cloud2}
	mergedCloud, err := icp.AlignPointClouds(ctx, clouds, config, logger)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, mergedCloud, test.ShouldNotBeNil)

	// Visualize aligned result
	if err := vizClient.DrawPointCloud("v_merged_result", mergedCloud, nil); err != nil {
		t.Logf("Warning: Failed to visualize: %v", err)
	}

	// Check that merged cloud has expected number of points
	expectedPoints := cloud1.Size() + cloud2.Size()
	test.That(t, mergedCloud.Size(), test.ShouldEqual, expectedPoints)

	// Verify that the aligned result is vertical (up-and-down)
	// Extract all points from merged cloud
	mergedOctree, err := pointcloud.ToBasicOctree(mergedCloud, 0)
	test.That(t, err, test.ShouldBeNil)
	mergedPoints := mergedOctree.ToPoints(0)
	test.That(t, len(mergedPoints), test.ShouldBeGreaterThan, 0)

	// Compute centroid and principal directions of merged cloud
	centroid := pcutils.ComputeCentroid(mergedPoints)
	principalDir, _, err := pcutils.ComputePrincipalDirections(mergedPoints, centroid)
	test.That(t, err, test.ShouldBeNil)

	// The principal direction should be mostly vertical (Z direction)
	// Check if principal direction is close to Z-axis (0, 0, 1) or (0, 0, -1)
	zAxis := r3.Vector{X: 0, Y: 0, Z: 1}
	dotProductZ := math.Abs(principalDir.Dot(zAxis))

	// The principal direction should be very close to vertical (dot product close to 1)
	test.That(t, dotProductZ, test.ShouldBeGreaterThan, 0.999)
}

// createVShapeRectangles creates two perfect flat rectangular point clouds that intersect to form a V shape.
func createVShapeRectangles(t *testing.T) (pointcloud.PointCloud, pointcloud.PointCloud) {
	// Rectangle dimensions
	width := 100.0
	length := 2000.0
	pointSpacing := 2.0

	vAngle := 85.0 * math.Pi / 180.0

	cloud1 := pointcloud.NewBasicEmpty()
	cloud2 := pointcloud.NewBasicEmpty()

	// Create first rectangle (angled upward)
	for x := -width / 2; x <= width/2; x += pointSpacing {
		for y := 0.0; y <= length; y += pointSpacing {
			// Rotate point by +vAngle around X axis
			z1 := y * math.Sin(vAngle)
			y1 := y * math.Cos(vAngle)

			point := r3.Vector{X: x, Y: y1, Z: z1}
			err := cloud1.Set(point, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}
	}

	// Create second rectangle (angled downward)
	for x := -width / 2; x <= width/2; x += pointSpacing {
		for y := 0.0; y <= length; y += pointSpacing {
			// Rotate point by -vAngle around X axis
			z2 := -y * math.Sin(vAngle)
			y2 := y * math.Cos(vAngle)

			point := r3.Vector{X: x, Y: y2, Z: z2 + 50}
			err := cloud2.Set(point, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}
	}

	return cloud1, cloud2
}

// Error condition tests.
func TestAlignPointCloudsFromFilesErrors(t *testing.T) {
	ctx := context.Background()

	config := icp.DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	t.Run("InsufficientFiles", func(t *testing.T) {
		_, err := icp.AlignPointCloudsFromFiles(ctx, []string{}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "need at least 1 file path, got 0")
	})

	t.Run("SingleFile", func(t *testing.T) {
		// Test that single file returns error for non-existent file
		_, err := icp.AlignPointCloudsFromFiles(ctx, []string{"nonexistent.pcd"}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "failed to load")
	})

	t.Run("InvalidFilePath", func(t *testing.T) {
		_, err := icp.AlignPointCloudsFromFiles(ctx, []string{"nonexistent1.pcd", "nonexistent2.pcd"}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "failed to load")
	})
}

func TestAlignPointCloudsErrors(t *testing.T) {
	ctx := context.Background()

	config := icp.DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	t.Run("InsufficientClouds", func(t *testing.T) {
		_, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "need at least 1 point cloud, got 0")
	})

	t.Run("SingleCloud", func(t *testing.T) {
		cloud := pointcloud.NewBasicEmpty()
		// Add some points to the cloud
		for i := 0; i < 10; i++ {
			err := cloud.Set(r3.Vector{X: float64(i), Y: 0, Z: 0}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}

		result, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{cloud}, config, logger)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, result, test.ShouldEqual, cloud) // Should return the same cloud
		test.That(t, result.Size(), test.ShouldEqual, cloud.Size())
	})

	t.Run("EmptyClouds", func(t *testing.T) {
		cloud1 := pointcloud.NewBasicEmpty()
		cloud2 := pointcloud.NewBasicEmpty()
		_, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{cloud1, cloud2}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
	})

	t.Run("InsufficientOverlap", func(t *testing.T) {
		// Create two clouds with no overlap
		cloud1 := pointcloud.NewBasicEmpty()
		cloud2 := pointcloud.NewBasicEmpty()

		// Add points to cloud1 in one region
		for i := 0; i < 50; i++ {
			err := cloud1.Set(r3.Vector{X: float64(i), Y: 0, Z: 0}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}

		// Add points to cloud2 in a distant region (no overlap)
		for i := 0; i < 50; i++ {
			err := cloud2.Set(r3.Vector{X: float64(i) + 1000, Y: 1000, Z: 1000}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}

		_, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{cloud1, cloud2}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "no valid overlaps found in graph")
	})

	t.Run("DisconnectedComponents", func(t *testing.T) {
		// Create 3 clouds where 2 overlap but the 3rd is isolated
		cloud1, cloud2 := createMinimalOverlappingClouds(t)
		cloud3 := pointcloud.NewBasicEmpty()

		// Add isolated cloud3 with no overlap
		for i := 0; i < 50; i++ {
			err := cloud3.Set(r3.Vector{X: float64(i) + 2000, Y: 2000, Z: 2000}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}

		_, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{cloud1, cloud2, cloud3}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "disconnected component detected, cannot merge")
	})
}

func TestConfigurationErrors(t *testing.T) {
	ctx := context.Background()

	logger := logging.NewTestLogger(t)
	cloud1, cloud2 := createMinimalOverlappingClouds(t)

	t.Run("NegativeOverlapThreshold", func(t *testing.T) {
		config := icp.DefaultICPConfig()
		config.OverlapThreshold = -1.0

		_, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{cloud1, cloud2}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
	})

	t.Run("ExcessiveMinOverlapPoints", func(t *testing.T) {
		config := icp.DefaultICPConfig()
		config.MinOverlapPoints = 10000 // More than any realistic cloud

		_, err := icp.AlignPointClouds(ctx, []pointcloud.PointCloud{cloud1, cloud2}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "no valid overlaps found")
	})
}

// Helper function to create minimal overlapping clouds for testing.
func createMinimalOverlappingClouds(t *testing.T) (pointcloud.PointCloud, pointcloud.PointCloud) {
	cloud1 := pointcloud.NewBasicEmpty()
	cloud2 := pointcloud.NewBasicEmpty()

	// Create overlapping regions with sufficient points
	for i := 0; i < 50; i++ {
		for j := 0; j < 50; j++ {
			// Cloud1 points
			err := cloud1.Set(r3.Vector{X: float64(i), Y: float64(j), Z: 0}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)

			// Cloud2 points with slight offset to create overlap
			err = cloud2.Set(r3.Vector{X: float64(i) + 5, Y: float64(j) + 5, Z: 1}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}
	}

	return cloud1, cloud2
}
