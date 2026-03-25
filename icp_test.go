package icp

// This file is full of tests that rely on the internal (non-exported) implementation of the ICP
// package. For black-box testing, see icp_blackbox_test.go.

import (
	"context"
	"math"
	"math/rand"
	"runtime/debug"
	"testing"

	"github.com/golang/geo/r3"
	vizClient "github.com/viam-labs/motion-tools/client/client"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/test"
)

func init() {
	// pointcloud merging uses a fair bit of memory, we need to not crash the github action runner
	debug.SetGCPercent(40)
}

func TestICPAlignmentSynthetic(t *testing.T) {
	ctx := context.Background()
	// Create deterministic random source for reproducible tests
	rng := rand.New(rand.NewSource(42)) //nolint:gosec

	// Create test pointclouds
	cloud1, cloud2 := createTestPointclouds(t, rng)

	// Visualize original point clouds before alignment
	err := vizClient.DrawPointCloud("cloud1_original", cloud1, &[3]uint8{0, 255, 0})
	if err != nil {
		t.Logf("Warning: Failed to visualize: %v", err)
	}
	err = vizClient.DrawPointCloud("cloud2_original", cloud2, &[3]uint8{0, 255, 255})
	if err != nil {
		t.Logf("Warning: Failed to visualize: %v", err)
	}
	config := DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	transform, oc1, oc2, err := computeAlignment(ctx, cloud1, cloud2, config, logger, nil)
	test.That(t, err, test.ShouldBeNil)

	// The pose here is not the raw pose output by nlopt, we get the centroid of the overlap region (i.e. what we aligned),
	// and its inverse, and construct a compose(centroidPose, alignmentPose, inverseCentroidPose) and that is what you see here. That is
	// because that is the pose that will then put the entire parent octree in the correct spot if you do octree1.Transform(transform)
	expectedTransform := spatialmath.NewPose(
		r3.Vector{X: -9.7, Y: 21.3, Z: 8.5},
		&spatialmath.OrientationVectorDegrees{OX: 0.017229, OY: 0.006477, OZ: 0.999831, Theta: -28.797890},
	)

	// Test asymmetric alignment (symmetrical = false)
	t.Run("AsymmetricAlignment", func(t *testing.T) {
		mergedCloud, err := applyTransformAndMerge(ctx, transform, oc1, oc2, false, logger)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, mergedCloud, test.ShouldNotBeNil)

		// Check that merged cloud has expected number of points
		expectedPoints := cloud1.Size() + cloud2.Size()
		test.That(t, mergedCloud.Size(), test.ShouldEqual, expectedPoints)

		// Verify that all points from cloud2 are preserved (reference cloud stays unchanged)
		cloud2Points := oc2.ToPoints(0)

		// Check that every point in cloud2 has an equivalent point in the merged cloud
		cloud2FoundCount := 0
		for _, cloud2Point := range cloud2Points {
			_, exists := mergedCloud.At(cloud2Point.X, cloud2Point.Y, cloud2Point.Z)
			if exists {
				cloud2FoundCount++
			}
		}
		test.That(t, cloud2FoundCount, test.ShouldEqual, len(cloud2Points))

		// Verify that original cloud1 points are NOT in the merged cloud (they should be transformed)
		cloud1Points := oc1.ToPoints(0)
		cloud1FoundCount := 0
		for _, cloud1Point := range cloud1Points {
			_, exists := mergedCloud.At(cloud1Point.X, cloud1Point.Y, cloud1Point.Z)
			if exists {
				cloud1FoundCount++
			}
		}
		// Original cloud1 points should not be found in merged cloud (they were transformed)
		test.That(t, cloud1FoundCount, test.ShouldEqual, 0)

		// Visualize asymmetric result
		err = vizClient.DrawPointCloud("merged_result_asymmetric", mergedCloud, nil)
		if err != nil {
			t.Logf("Warning: Failed to visualize: %v", err)
		}
	})

	// Test symmetric alignment (symmetrical = true)
	t.Run("SymmetricAlignment", func(t *testing.T) {
		mergedCloud, err := applyTransformAndMerge(ctx, transform, oc1, oc2, true, logger)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, mergedCloud, test.ShouldNotBeNil)

		// Check that merged cloud has expected number of points
		expectedPoints := cloud1.Size() + cloud2.Size()
		test.That(t, mergedCloud.Size(), test.ShouldEqual, expectedPoints)

		// In symmetric mode, both clouds are transformed, so neither original cloud should be preserved exactly
		cloud1Points := oc1.ToPoints(0)
		cloud2Points := oc2.ToPoints(0)

		// Check that original cloud1 points are NOT in the merged cloud
		cloud1FoundCount := 0
		for _, cloud1Point := range cloud1Points {
			_, exists := mergedCloud.At(cloud1Point.X, cloud1Point.Y, cloud1Point.Z)
			if exists {
				cloud1FoundCount++
			}
		}
		test.That(t, cloud1FoundCount, test.ShouldEqual, 0)

		// Check that original cloud2 points are NOT in the merged cloud
		cloud2FoundCount := 0
		for _, cloud2Point := range cloud2Points {
			_, exists := mergedCloud.At(cloud2Point.X, cloud2Point.Y, cloud2Point.Z)
			if exists {
				cloud2FoundCount++
			}
		}
		test.That(t, cloud2FoundCount, test.ShouldEqual, 0)

		// Visualize symmetric result
		err = vizClient.DrawPointCloud("merged_result_symmetric", mergedCloud, nil)
		if err != nil {
			t.Logf("Warning: Failed to visualize: %v", err)
		}
	})

	// Verify transform is close to expected. Note it will move around slightly.
	test.That(t, expectedTransform.Point().Distance(transform.Point()), test.ShouldBeLessThan, 2.)
	expectedOV := expectedTransform.Orientation().OrientationVectorDegrees()
	actualOV := transform.Orientation().OrientationVectorDegrees()
	test.That(t, math.Abs(expectedOV.Theta-actualOV.Theta), test.ShouldBeLessThan, 3.)
}

// Helper function to create test pointclouds that simulate camera-captured curved surface sections.
func createTestPointclouds(t *testing.T, rng *rand.Rand) (pointcloud.PointCloud, pointcloud.PointCloud) {
	// Create two overlapping sections of the same surface
	// Both cameras see the same region but camera2 is misaligned
	thetaStart, thetaEnd := 0.0, math.Pi
	zStart, zEnd := 0.0, 800.0

	points1 := createCameraViewOfSurface(thetaStart, 3.0, 10.0, zEnd, 15000, rng)
	points2 := createCameraViewOfSurface(0.1, thetaEnd, zStart, zEnd+100, 16000, rng)

	// Apply misalignment to camera2 points
	transform := spatialmath.NewPose(
		r3.Vector{X: 0, Y: 0, Z: 300},
		&spatialmath.OrientationVector{OY: 0.1, OZ: 1, Theta: 2},
	)

	// Create pointclouds
	cloud1 := pointcloud.NewBasicEmpty()
	for _, p := range points1 {
		err := cloud1.Set(p, pointcloud.NewBasicData())
		test.That(t, err, test.ShouldBeNil)
	}

	cloud2 := pointcloud.NewBasicEmpty()
	for _, p := range points2 {
		tformP := spatialmath.Compose(transform, spatialmath.NewPoseFromPoint(p)).Point()
		err := cloud2.Set(tformP, pointcloud.NewBasicData())
		test.That(t, err, test.ShouldBeNil)
	}

	return cloud1, cloud2
}

// Create a realistic camera view of a curved surface section.
// Note that this is duplicated in icp_blackbox_test.go.
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
// Note that this is duplicated in icp_blackbox_test.go.
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

// Unit tests for individual functions.
func TestFindNearestPoint(t *testing.T) {
	target := r3.Vector{1, 1, 1}

	// Test empty slice
	_, dist := findNearestPoint(target, []r3.Vector{})
	test.That(t, math.IsInf(dist, 1), test.ShouldBeTrue)

	// Test single point
	points := []r3.Vector{{2, 2, 2}}
	nearest, dist := findNearestPoint(target, points)
	expectedDist := math.Sqrt(3) // distance from (1,1,1) to (2,2,2)
	test.That(t, dist, test.ShouldAlmostEqual, expectedDist)
	test.That(t, nearest, test.ShouldResemble, points[0])

	// Test multiple points
	points = []r3.Vector{{0, 0, 0}, {1.1, 1.1, 1.1}, {5, 5, 5}}
	nearest, _ = findNearestPoint(target, points)
	expected := r3.Vector{1.1, 1.1, 1.1}
	test.That(t, nearest, test.ShouldResemble, expected)
}

func TestComputeAlignmentErrors(t *testing.T) {
	ctx := context.Background()

	config := DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	t.Run("InsufficientOverlapPoints", func(t *testing.T) {
		// Create clouds with minimal overlap below threshold
		cloud1 := pointcloud.NewBasicEmpty()
		cloud2 := pointcloud.NewBasicEmpty()

		// Create very small overlap (less than MinOverlapPoints)
		for i := 0; i < 5; i++ {
			err := cloud1.Set(r3.Vector{X: float64(i), Y: 0, Z: 0}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
			err = cloud2.Set(r3.Vector{X: float64(i) + 0.1, Y: 0.1, Z: 0.1}, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}

		_, _, _, err := computeAlignment(ctx, cloud1, cloud2, config, logger, nil)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "insufficient overlap")
	})

	t.Run("InsufficientPointsForOptimization", func(t *testing.T) {
		// Create minimal test case that passes overlap check but fails optimization
		cloud1 := pointcloud.NewBasicEmpty()
		cloud2 := pointcloud.NewBasicEmpty()

		// Add exactly MinOverlapPoints but not enough for NLOpt (< 3)
		overlap := overlapRegion{
			query:     []r3.Vector{{0, 0, 0}, {1, 0, 0}},
			reference: []r3.Vector{{0, 0, 0}, {1, 0, 0}},
		}

		for _, p := range overlap.query {
			err := cloud1.Set(p, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}
		for _, p := range overlap.reference {
			err := cloud2.Set(p, pointcloud.NewBasicData())
			test.That(t, err, test.ShouldBeNil)
		}

		// Set low MinOverlapPoints to bypass first check
		lowConfig := config
		lowConfig.MinOverlapPoints = 2

		_, _, _, err := computeAlignment(ctx, cloud1, cloud2, lowConfig, logger, &overlap)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "insufficient points for NLOpt optimization")
	})
}

func TestBuildOverlapGraphErrors(t *testing.T) {
	ctx := context.Background()

	config := DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	t.Run("InsufficientClouds", func(t *testing.T) {
		_, err := buildOverlapGraph(ctx, []pointcloud.PointCloud{}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "need at least 2 point clouds, got 0")

		cloud := pointcloud.NewBasicEmpty()
		_, err = buildOverlapGraph(ctx, []pointcloud.PointCloud{cloud}, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "need at least 2 point clouds, got 1")
	})
}

func TestSelectMergeOrderErrors(t *testing.T) {
	t.Run("EmptyGraph", func(t *testing.T) {
		graph := &overlapGraph{
			cloudCount: 3,
			edges:      []overlapEdge{},
			edgeMap:    make(map[string]*overlapEdge),
		}

		// Mock centroids for the 3 clouds
		centroids := []r3.Vector{
			{X: 0, Y: 0, Z: 0},
			{X: 10, Y: 10, Z: 0},
			{X: 20, Y: 0, Z: 0},
		}

		_, err := selectMergeOrder(graph, centroids)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "no valid overlaps found in graph")
	})

	t.Run("DisconnectedComponents", func(t *testing.T) {
		// Create graph with disconnected components
		graph := &overlapGraph{
			cloudCount: 4,
			edges: []overlapEdge{
				{cloudIndex1: 0, cloudIndex2: 1, overlapPoints: 100},
				// Cloud 2 and 3 are disconnected from 0,1
				{cloudIndex1: 2, cloudIndex2: 3, overlapPoints: 50},
			},
			edgeMap: make(map[string]*overlapEdge),
		}

		// Mock centroids for the 4 clouds
		centroids := []r3.Vector{
			{X: 0, Y: 0, Z: 0},
			{X: 5, Y: 0, Z: 0},
			{X: 100, Y: 100, Z: 0},
			{X: 105, Y: 100, Z: 0},
		}

		_, err := selectMergeOrder(graph, centroids)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "disconnected component detected")
	})

	t.Run("MismatchedCentroidCount", func(t *testing.T) {
		graph := &overlapGraph{
			cloudCount: 3,
			edges: []overlapEdge{
				{cloudIndex1: 0, cloudIndex2: 1, overlapPoints: 100},
			},
			edgeMap: make(map[string]*overlapEdge),
		}

		// Provide wrong number of centroids (2 instead of 3)
		centroids := []r3.Vector{
			{X: 0, Y: 0, Z: 0},
			{X: 10, Y: 10, Z: 0},
		}

		_, err := selectMergeOrder(graph, centroids)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "cloudCentroids length (2) does not match cloudCount (3)")
	})
}

func TestAlignOverlapRegionErrors(t *testing.T) {
	ctx := context.Background()

	config := DefaultICPConfig()
	logger := logging.NewTestLogger(t)

	t.Run("InsufficientCorrespondences", func(t *testing.T) {
		// Create overlap with insufficient correspondences for optimization
		overlap := overlapRegion{
			query:     []r3.Vector{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}},
			reference: []r3.Vector{{100, 100, 100}, {101, 100, 100}, {101, 100, 101}}, // Far away, no correspondences
		}

		_, err := alignOverlapRegion(ctx, overlap, config, logger)
		test.That(t, err, test.ShouldNotBeNil)
		test.That(t, err.Error(), test.ShouldContainSubstring, "insufficient correspondences")
	})
}
