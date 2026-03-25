package icp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan/ik"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/utils"
	"gonum.org/v1/gonum/num/quat"

	"github.com/viamrobotics/icp/pcutils"
)

// Default configuration values for ICP alignment.
var (
	// defaultOverlapThreshold is the default distance threshold for considering points as overlapping (mm).
	defaultOverlapThreshold = 25.0

	// defaultMinOverlapPoints is the default minimum number of overlapping points required to align two pointclouds.
	defaultMinOverlapPoints = 30

	// defaultMaxIterations is the default maximum number of ICP nlopt optimization iterations.
	defaultMaxIterations = 100

	// defaultConvergenceEpsilon is the default convergence threshold (mm).
	defaultConvergenceEpsilon = 0.001

	defaultFinalRealign = false

	// Thread management for parallelization.
	maxMultiAlignmentWorkers = 4 // Workers for multi-cloud overlap graph computation
	maxMutualNNThreads       = 4 // Threads for mutual nearest neighbor finding

	// Downsample the pointcloud to be aligned to one point per this many mm-square voxel.
	voxelSize = 5.
)

// ICPConfig contains parameters for controlling the ICP alignment process.
type ICPConfig struct {
	// OverlapThreshold is the maximum distance (in mm) for considering points as overlapping.
	// Smaller values require tighter overlap but may be more accurate.
	OverlapThreshold float64

	// MinOverlapPoints is the minimum number of overlapping points required for alignment.
	// Insufficient overlap will cause alignment to fail with an error.
	MinOverlapPoints int

	// MaxIterations is the maximum number of ICP optimization iterations.
	// More iterations may improve accuracy but increase computation time.
	MaxIterations int

	// ConvergenceEpsilon is the convergence threshold in mm.
	// Optimization stops when improvement falls below this value.
	ConvergenceEpsilon float64

	// Run a realignment at the very end against the sum total of all unaligned pointclouds.
	// May be very slow
	FinalRealign bool
}

// DefaultICPConfig returns a configuration with sensible defaults for sub-mm accuracy.
// These defaults are suitable for most point cloud alignment tasks involving millimeter-scale precision.
func DefaultICPConfig() ICPConfig {
	return ICPConfig{
		OverlapThreshold:   defaultOverlapThreshold,
		MinOverlapPoints:   defaultMinOverlapPoints,
		MaxIterations:      defaultMaxIterations,
		ConvergenceEpsilon: defaultConvergenceEpsilon,
		FinalRealign:       defaultFinalRealign,
	}
}

type overlapRegion struct {
	query     []r3.Vector // Points from cloud1 in overlap region
	reference []r3.Vector // Points from cloud2 in overlap region
}

// These below are used to determine which pointclouds should be merged together in the event many are passed in
// overlapEdge represents an overlap connection between two point clouds.
type overlapEdge struct {
	cloudIndex1   int
	cloudIndex2   int
	overlapPoints int           // Number of overlapping points
	overlapRegion overlapRegion // The actual overlap data
}

// overlapGraph represents the connectivity between multiple point clouds.
type overlapGraph struct {
	cloudCount int                     // Number of point clouds
	edges      []overlapEdge           // All overlap connections
	edgeMap    map[string]*overlapEdge // Quick lookup: "i,j" -> edge
}

// AlignPointCloudsFromFiles loads point clouds from files and performs ICP alignment.
// This is a convenience function that combines file loading with alignment.
// It requires at least 2 file paths and returns transforms for each cloud and the merged result.
// For 2 clouds, transforms are symmetric. For 3+ clouds, the first two are merged symmetrically,
// then remaining clouds are aligned to the growing reference.
func AlignPointCloudsFromFiles(ctx context.Context, filePaths []string, config ICPConfig, logger logging.Logger) (pointcloud.PointCloud, error) {
	if len(filePaths) == 0 {
		return nil, fmt.Errorf("need at least 1 file path, got 0")
	}

	clouds := make([]pointcloud.PointCloud, len(filePaths))
	for i, filePath := range filePaths {
		cloud, err := pointcloud.NewFromFile(filePath, "")
		if err != nil {
			return nil, fmt.Errorf("failed to load %s: %w", filePath, err)
		}
		clouds[i] = cloud
	}

	return AlignPointClouds(ctx, clouds, config, logger)
}

// AlignPointClouds performs Iterative Closest Point alignment between point clouds and returns the merged result.
// This is the main function for point cloud alignment in production code.
//
// Parameters:
//   - clouds: slice of point clouds to align (minimum 2 required)
//   - config: ICP configuration parameters
//   - logger: logger for error and warning messages
//
// Returns:
//   - pointcloud.PointCloud: merged result of all aligned clouds
//   - error: any errors during alignment process
//
// For 2 clouds: returns symmetric transformations that move both clouds toward each other.
// For 3+ clouds: first two clouds to merge get symmetric transforms, remaining clouds align to the reference.
// The function will return an error if clouds cannot be aligned.
func AlignPointClouds(ctx context.Context, clouds []pointcloud.PointCloud, config ICPConfig, logger logging.Logger) (pointcloud.PointCloud, error) {
	if len(clouds) == 0 {
		return nil, fmt.Errorf("need at least 1 point cloud, got 0")
	}

	// If only one cloud, return it directly
	if len(clouds) == 1 {
		return clouds[0], nil
	}

	logger.Debugf("ICP Pipeline: Starting alignment of %d pointclouds (sizes: %v)",
		len(clouds), func() []int {
			sizes := make([]int, len(clouds))
			for i, cloud := range clouds {
				sizes[i] = cloud.Size()
			}
			return sizes
		}())

	// Build overlap graph
	logger.Debug("ICP Pipeline Stage 1: Building overlap graph between all pointcloud pairs")
	graph, err := buildOverlapGraph(ctx, clouds, config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to build overlap graph: %w", err)
	}
	logger.Debugf("ICP Pipeline Stage 1: Found %d valid overlap edges", len(graph.edges))

	// Calculate centroids of all clouds for later use
	// We use this to determine merge order; the most central cloud is the one used as the seed from which we expand outwards.
	cloudCentroids := make([]r3.Vector, len(clouds))
	for i, cloud := range clouds {
		points := pointcloud.CloudToPoints(cloud)
		if len(points) == 0 {
			return nil, fmt.Errorf("cloud %d is empty", i)
		}
		cloudCentroids[i] = pcutils.ComputeCentroid(points)
	}

	// Select merge order
	logger.Debug("ICP Pipeline Stage 2: Selecting optimal merge order")
	mergePairs, err := selectMergeOrder(graph, cloudCentroids)
	if err != nil {
		return nil, err
	}
	logger.Debugf("ICP Pipeline Stage 2: Selected merge sequence with %d pairs", len(mergePairs))
	for i, edge := range mergePairs {
		logger.Debugf("  Merge %d: Cloud %d -> Cloud %d (overlap: %d points)",
			i+1, edge.cloudIndex1, edge.cloudIndex2, edge.overlapPoints)
	}

	// Perform the first merge (symmetric alignment between two clouds)
	firstEdge := mergePairs[0]
	cloud1 := clouds[firstEdge.cloudIndex1]
	cloud2 := clouds[firstEdge.cloudIndex2]

	logger.Debugf("ICP Pipeline Stage 3: Performing initial symmetric merge of clouds %d and %d",
		firstEdge.cloudIndex1, firstEdge.cloudIndex2)
	logger.Debugf("  Cloud %d size: %d points, Cloud %d size: %d points",
		firstEdge.cloudIndex1, cloud1.Size(), firstEdge.cloudIndex2, cloud2.Size())

	// Use pre-computed overlap from graph construction to avoid recalculating
	// Align the first two pointclouds to one another to create our reference
	mergedReferenceInit, err := alignTwoPointClouds(ctx, cloud1, cloud2, config, logger, &firstEdge.overlapRegion, true)
	if err != nil {
		return nil, err
	}
	// Merge the two original pointclouds together without aligning
	naiveMerge, err := NaiveMergeClouds(ctx, cloud1, cloud2, logger)
	if err != nil {
		return nil, err
	}
	// Realign the references to the entire two original clouds, to best place it and minimize errors.
	transform, refOctree, _, err := computeAlignment(ctx, mergedReferenceInit, naiveMerge, config, logger, nil)
	if err != nil {
		return nil, err
	}
	mergedReference := refOctree.Transform(transform).(pointcloud.PointCloud)
	logger.Debugf("ICP Pipeline Stage 3: Initial merge completed, merged cloud size: %d points", mergedReference.Size())

	// Track which clouds have been merged into the reference
	mergedIntoReference := make([]bool, len(clouds))
	mergedIntoReference[firstEdge.cloudIndex1] = true
	mergedIntoReference[firstEdge.cloudIndex2] = true

	// Perform subsequent merges (new cloud aligns to growing reference)
	for i, edge := range mergePairs[1:] {
		// Determine which cloud index to merge (the one not already merged)
		var newCloudIndex int
		switch {
		case mergedIntoReference[edge.cloudIndex1] && !mergedIntoReference[edge.cloudIndex2]:
			newCloudIndex = edge.cloudIndex2
		case mergedIntoReference[edge.cloudIndex2] && !mergedIntoReference[edge.cloudIndex1]:
			newCloudIndex = edge.cloudIndex1
		default:
			return nil, fmt.Errorf("invalid merge state for edge %d-%d", edge.cloudIndex1, edge.cloudIndex2)
		}

		if newCloudIndex >= len(clouds) {
			return nil, fmt.Errorf("invalid cloud index: %d", newCloudIndex)
		}

		newCloud := clouds[newCloudIndex]
		naiveMerge, err = NaiveMergeClouds(ctx, naiveMerge, newCloud, logger)
		if err != nil {
			return nil, err
		}
		logger.Debugf("ICP Pipeline Stage 4.%d: Aligning cloud %d (size: %d) to merged reference (size: %d)",
			i+1, newCloudIndex, newCloud.Size(), mergedReference.Size())

		// Align new cloud to the current merged reference
		updatedReference, err := alignTwoPointClouds(ctx, newCloud, mergedReference, config, logger, nil, false)
		if err != nil {
			return nil, fmt.Errorf("failed to align cloud %d: %w", newCloudIndex, err)
		}

		mergedIntoReference[newCloudIndex] = true

		// Update the merged reference
		mergedReference = updatedReference
		logger.Debugf("ICP Pipeline Stage 4.%d: Cloud %d merged, updated reference size: %d points",
			i+1, newCloudIndex, mergedReference.Size())
	}
	if config.FinalRealign {
		// Finally, align the merged cloud to the sum total of the original unaligned clouds
		// This is moderately expensive and it's not clear that it is very useful after the initial realignment is done
		// TODO: eventually enable or remove this once its utility is apparent, or at least make it more easily configurable.
		transform, refOctree, _, err = computeAlignment(ctx, mergedReference, naiveMerge, config, logger, nil)
		if err != nil {
			return nil, err
		}
		mergedReference = refOctree.Transform(transform).(*pointcloud.BasicOctree)
	}

	// Check if all clouds were successfully merged
	unmergedCount := 0
	for _, merged := range mergedIntoReference {
		if !merged {
			unmergedCount++
		}
	}

	if unmergedCount > 0 {
		return nil, fmt.Errorf("%d clouds remain unmerged due to insufficient overlap (disconnected components)", unmergedCount)
	}

	logger.Debugf("ICP Pipeline Complete: Successfully merged %d pointclouds into final result with %d points",
		len(clouds), mergedReference.Size())
	return mergedReference, nil
}

func alignTwoPointClouds(
	ctx context.Context,
	cloud1, cloud2 pointcloud.PointCloud,
	config ICPConfig,
	logger logging.Logger,
	precomputedOverlap *overlapRegion,
	symmetrical bool,
) (pointcloud.PointCloud, error) {
	transform, octree1, octree2, err := computeAlignment(ctx, cloud1, cloud2, config, logger, precomputedOverlap)
	if err != nil {
		return nil, err
	}
	return applyTransformAndMerge(ctx, transform, octree1, octree2, symmetrical, logger)
}

func computeAlignment(
	ctx context.Context,
	cloud1, cloud2 pointcloud.PointCloud,
	config ICPConfig,
	logger logging.Logger,
	precomputedOverlap *overlapRegion,
) (spatialmath.Pose, *pointcloud.BasicOctree, *pointcloud.BasicOctree, error) {
	var overlap overlapRegion
	var err error

	octree1, err := pointcloud.ToBasicOctree(cloud1, 0)
	if err != nil {
		return nil, nil, nil, err
	}
	octree2, err := pointcloud.ToBasicOctree(cloud2, 0)
	if err != nil {
		return nil, nil, nil, err
	}

	// Generate the overlap if it wasn't passed in
	if precomputedOverlap != nil {
		// Use pre-computed overlap if passed (first merge). Do not use after the first merge, as the reference is not in the same place.
		overlap = *precomputedOverlap
		logger.Debugf("  Using precomputed overlap: %d query points, %d reference points",
			len(overlap.query), len(overlap.reference))
	} else {
		logger.Debug("  Computing overlap region between pointclouds")
		// Find overlap region
		overlap, err = findOverlapRegion(octree1, octree2, config)
		if err != nil {
			return nil, nil, nil, err
		}
		logger.Debugf("  Found overlap region: %d query points, %d reference points",
			len(overlap.query), len(overlap.reference))
	}

	if len(overlap.query) < config.MinOverlapPoints {
		return nil, nil, nil, fmt.Errorf("insufficient overlap: only %d points found (minimum %d required)",
			len(overlap.query), config.MinOverlapPoints)
	}

	// Run ICP on overlap region
	logger.Debug("  Starting ICP optimization on overlap region")
	finalTransform, err := alignOverlapRegion(ctx, overlap, config, logger)
	if err != nil {
		return nil, nil, nil, err
	}
	logger.Debugf("  ICP optimization completed, transform: %v", finalTransform)

	return finalTransform, octree1, octree2, nil
}

func applyTransformAndMerge(
	ctx context.Context,
	transform spatialmath.Pose,
	octree1, octree2 *pointcloud.BasicOctree,
	symmetrical bool,
	logger logging.Logger,
) (pointcloud.PointCloud, error) {
	var transformedCloud1 pointcloud.PointCloud
	transformedCloud2 := octree2
	if symmetrical {
		// Split transformation and apply symmetrically to both clouds
		// We optimize the transformation of one to the other, but the reality will actually be in the middle, since both clouds
		// were (presumably) produced with the same error.
		halfwayTransform := spatialmath.Interpolate(spatialmath.NewZeroPose(), transform, 0.5)
		transform1 := halfwayTransform
		transform2 := spatialmath.PoseInverse(halfwayTransform)

		logger.Debug("  Applying symmetric transforms and merging pointclouds")
		transformedCloud1 = octree1.Transform(transform1).(*pointcloud.BasicOctree)
		transformedCloud2 = octree2.Transform(transform2).(*pointcloud.BasicOctree)
	} else {
		transformedCloud1 = octree1.Transform(transform).(*pointcloud.BasicOctree)
	}
	return NaiveMergeClouds(ctx, transformedCloud1, transformedCloud2, logger)
}

// NaiveMergeClouds concatenates two point clouds without alignment.
func NaiveMergeClouds(ctx context.Context, cloud1, cloud2 pointcloud.PointCloud, logger logging.Logger) (pointcloud.PointCloud, error) {
	mergedCloud := pointcloud.NewBasicEmpty()
	func1 := func(context context.Context) (pointcloud.PointCloud, spatialmath.Pose, error) {
		return cloud1, nil, nil
	}
	func2 := func(context context.Context) (pointcloud.PointCloud, spatialmath.Pose, error) {
		return cloud2, nil, nil
	}
	err := pointcloud.MergePointClouds(ctx, []pointcloud.CloudAndOffsetFunc{func1, func2}, mergedCloud)
	if err != nil {
		return nil, err
	}
	logger.Debugf("  Pointclouds merged: %d + %d = %d points",
		cloud1.Size(), cloud2.Size(), mergedCloud.Size())

	return mergedCloud, nil
}

func findOverlapRegion(octree1, octree2 *pointcloud.BasicOctree, config ICPConfig) (overlapRegion, error) {
	overlap := overlapRegion{}
	downsampledCloud1 := pcutils.VoxelDownsample(octree1.ToPoints(0), voxelSize)
	overlapOctree, err := pcutils.VectorsToOctree(downsampledCloud1)
	if err != nil {
		return overlapRegion{}, err
	}

	overlap.reference = octree2.PointsCollidingWith([]spatialmath.Geometry{overlapOctree}, config.OverlapThreshold)
	refOctree, err := pcutils.VectorsToOctree(overlap.reference)
	if err != nil {
		return overlapRegion{}, err
	}

	// We're going to find the closest points in parallel. Results will be put in resultChan, which
	// will be collected into a list once all points are finished.
	numThreads := min(maxMutualNNThreads, len(downsampledCloud1))
	workers := pcutils.NewParallelWorkers(numThreads)
	resultChan := make(chan r3.Vector, len(downsampledCloud1))

	for _, p1 := range downsampledCloud1 {
		workers.Do(func() {
			// Find nearest neighbors in cloud2
			nearbyPoints2, err := refOctree.PointsWithinRadius(p1, config.OverlapThreshold)
			if err != nil || len(nearbyPoints2) == 0 {
				return
			}

			closest, minDist := findNearestPoint(p1, nearbyPoints2)

			if minDist <= config.OverlapThreshold {
				// Verify mutual nearest neighbor relationship
				nearbyPoints1, err := octree1.PointsWithinRadius(closest, config.OverlapThreshold)
				if err != nil {
					// Only include points where the two points are each other's nearest neighbor
					return
				}

				if len(nearbyPoints1) > 0 {
					mutualClosest, _ := findNearestPoint(closest, nearbyPoints1)

					// If p1 is the closest to 'closest', then it's a mutual match
					if p1.Distance(mutualClosest) < 1e-6 {
						resultChan <- p1
					}
				}
			}
		})
	}
	workers.Wait()
	close(resultChan)

	for matchedPoint := range resultChan {
		overlap.query = append(overlap.query, matchedPoint)
	}

	return overlap, nil
}

// Find nearest point in a set of points.
func findNearestPoint(target r3.Vector, points []r3.Vector) (r3.Vector, float64) {
	if len(points) == 0 {
		return r3.Vector{}, math.Inf(1)
	}

	nearest := points[0]
	minDist := target.Distance(nearest)

	for _, p := range points[1:] {
		if dist := target.Distance(p); dist < minDist {
			nearest = p
			minDist = dist
		}
	}

	return nearest, minDist
}

// alignOverlapRegion performs ICP alignment using NLOpt optimization with centroid-relative coordinates.
func alignOverlapRegion(ctx context.Context, overlap overlapRegion, config ICPConfig, logger logging.Logger) (spatialmath.Pose, error) {
	logger.Debugf("    ICP Optimization: %d source points, %d target points",
		len(overlap.query), len(overlap.reference))

	// Compute centroid of query points (cloud1) and create centroid pose
	queryCentroid := pcutils.ComputeCentroid(overlap.query)
	centroidPose := spatialmath.NewPoseFromPoint(queryCentroid)
	logger.Debugf("    Query centroid: [%.2f, %.2f, %.2f]",
		queryCentroid.X, queryCentroid.Y, queryCentroid.Z)

	// Convert query points to centroid-relative coordinates
	sourcePoints := make([]r3.Vector, len(overlap.query)) // Points from cloud1 (moving) relative to centroid
	for i, p := range overlap.query {
		sourcePoints[i] = p.Sub(queryCentroid)
	}

	if len(sourcePoints) < 3 || len(overlap.reference) < 3 {
		return nil, fmt.Errorf(
			"insufficient points for NLOpt optimization: need at least 3 points, got source=%d target=%d",
			len(sourcePoints),
			len(overlap.reference),
		)
	}

	// Build target octree for correspondence finding
	targetOctree, err := pcutils.VectorsToOctree(overlap.reference)
	if err != nil {
		return nil, err
	}

	// Pre-compute correspondences using world-coordinate query points
	correspondences := preComputeCorrespondences(overlap.query, targetOctree, config.OverlapThreshold)
	if len(correspondences) < 3 {
		return nil, fmt.Errorf("insufficient correspondences for NLOpt optimization: need at least 3, got %d", len(correspondences))
	}
	logger.Debugf("    Found %d correspondences for optimization", len(correspondences))

	// Define optimization parameters: [tx, ty, tz, qx, qy, qz, qw]
	// Small bounds for centroid-relative optimization
	limits := []referenceframe.Limit{
		{Min: -10, Max: 10},
		{Min: -10, Max: 10},
		{Min: -10, Max: 10},
		{Min: -1, Max: 1},
		{Min: -1, Max: 1},
		{Min: -1, Max: 1},
		{Min: -1, Max: 1},
	}

	// Create NLOpt solver
	solver, err := ik.CreateNloptSolver(limits, logger, config.MaxIterations, false, true)
	if err != nil {
		return nil, err
	}

	// Create objective function
	objectiveFunc := createICPObjectiveFunction(overlap.query, correspondences, queryCentroid)

	// Initial guess: near zero pose
	// We don't want to use the true zero pose here. Reason being, nlopt will actually return an error of generic "FAILURE" if it is
	// unable to improve on the provided seed despite having nonzero score. Providing a small, random translation to the start guess
	// guarantees we are able to improve on the initial guess without masking other failures of this sort.
	rng := rand.New(rand.NewSource(42)) //nolint:gosec
	initialGuess := []float64{1 + rng.Float64(), 1 + rng.Float64(), rng.Float64(), 0, 0, 0, 1}

	// Run optimization
	logger.Debugf("    Starting NLOpt optimization with max %d iterations", config.MaxIterations)
	solutionChan := make(chan *ik.Solution, 10)
	var optErr error
	utils.PanicCapturingGo(func() {
		defer close(solutionChan)
		optErr = solver.Solve(ctx, solutionChan, initialGuess, 0, 0, objectiveFunc, 42)
	})

	// Collect best solution
	var bestSolution *ik.Solution
	for solution := range solutionChan {
		if bestSolution == nil || solution.Score < bestSolution.Score {
			bestSolution = solution
		}
	}
	// Transient errors (which can happen from nlopt due to numeric coincidences failing to converge) do not matter so long as it was also
	// able to return at least one solution.
	if optErr != nil && bestSolution == nil {
		return nil, optErr
	}

	if bestSolution == nil {
		return nil, errors.New("NLOpt ICP found no solution")
	}
	logger.Debugf("    Optimization converged with score: %.6f", bestSolution.Score)

	// Convert solution to pose
	params := bestSolution.Configuration
	translation := r3.Vector{X: params[0], Y: params[1], Z: params[2]}
	// Create quaternion from normalized components
	quatNum := quat.Number{
		Real: params[6], // qw
		Imag: params[3], // qx
		Jmag: params[4], // qy
		Kmag: params[5], // qz
	}
	quatNum = spatialmath.Normalize(quatNum)
	orientation := &spatialmath.Quaternion{
		Real: quatNum.Real,
		Imag: quatNum.Imag,
		Jmag: quatNum.Jmag,
		Kmag: quatNum.Kmag,
	}
	centroidRelativeTransform := spatialmath.NewPose(translation, orientation)

	// Transform the result back to world space
	// The final world transformation is: centroidPose.Compose(centroidRelativeTransform).Compose(centroidPose.Inverse())
	centroidInverse := spatialmath.PoseInverse(centroidPose)
	worldTransform := spatialmath.Compose(centroidPose, spatialmath.Compose(centroidRelativeTransform, centroidInverse))

	return worldTransform, nil
}

// correspondenceNlopt represents a correspondence for NLOpt optimization.
type correspondenceNlopt struct {
	sourceIdx int
	targetPts []r3.Vector
}

func preComputeCorrespondences(source []r3.Vector, targetOctree *pointcloud.BasicOctree, searchRadius float64) []correspondenceNlopt {
	correspondences := make([]correspondenceNlopt, 0, len(source))

	// For each source point, find all target points within radius
	for i, srcPoint := range source {
		neighbors, err := targetOctree.PointsWithinRadius(srcPoint, searchRadius)
		if err != nil || len(neighbors) == 0 {
			continue // Skip points without correspondences
		}

		correspondences = append(correspondences, correspondenceNlopt{
			sourceIdx: i,
			targetPts: neighbors,
		})
	}

	return correspondences
}

func createICPObjectiveFunction(worldQueryPoints []r3.Vector, correspondences []correspondenceNlopt, queryCentroid r3.Vector) func([]float64) float64 {
	return func(params []float64) float64 {
		// Extract transformation parameters (relative to centroid)
		translation := r3.Vector{X: params[0], Y: params[1], Z: params[2]}

		// Create quaternion from components and normalize
		quatNum := quat.Number{
			Real: params[6], // qw
			Imag: params[3], // qx
			Jmag: params[4], // qy
			Kmag: params[5], // qz
		}

		// Normalize quaternion to ensure valid rotation
		quatNum = spatialmath.Normalize(quatNum)

		// Create orientation from normalized quaternion
		orientation := &spatialmath.Quaternion{Real: quatNum.Real, Imag: quatNum.Imag, Jmag: quatNum.Jmag, Kmag: quatNum.Kmag}

		centroidRelativeTransform := spatialmath.NewPose(translation, orientation)

		// Compose the full world transformation: centroid -> relative transform -> inverse centroid
		centroidPose := spatialmath.NewPoseFromPoint(queryCentroid)
		centroidInverse := spatialmath.PoseInverse(centroidPose)
		worldTransform := spatialmath.Compose(centroidPose, spatialmath.Compose(centroidRelativeTransform, centroidInverse))

		var sumSquaredDist float64

		for _, corr := range correspondences {
			srcPoint := worldQueryPoints[corr.sourceIdx]

			// Apply world transformation to source point
			srcPose := spatialmath.NewPoseFromPoint(srcPoint)
			transformedPose := spatialmath.Compose(worldTransform, srcPose)
			worldTransformedPoint := transformedPose.Point()

			// Find closest point among pre-computed candidates
			minDist := math.Inf(1)
			for _, targetPt := range corr.targetPts {
				dist := worldTransformedPoint.Sub(targetPt).Norm()
				if dist < minDist {
					minDist = dist
				}
			}
			sumSquaredDist += minDist * minDist
		}

		return math.Sqrt(sumSquaredDist / float64(len(correspondences)))
	}
}

// selectMergeOrder determines the order in which to merge point clouds
// Calculate the centroid of each cloud, then the centroid-of-centroids,
// and start with the cloud whose centroid is closest to that centroid-of-centroids.
func selectMergeOrder(graph *overlapGraph, cloudCentroids []r3.Vector) ([]*overlapEdge, error) {
	if len(graph.edges) == 0 {
		return nil, fmt.Errorf("no valid overlaps found in graph")
	}

	if len(cloudCentroids) != graph.cloudCount {
		return nil, fmt.Errorf("cloudCentroids length (%d) does not match cloudCount (%d)", len(cloudCentroids), graph.cloudCount)
	}

	var mergeOrder []*overlapEdge
	merged := make([]bool, graph.cloudCount) // Track which clouds have been merged

	// Calculate centroid-of-centroids
	var centroidSum r3.Vector
	for _, centroid := range cloudCentroids {
		centroidSum = centroidSum.Add(centroid)
	}
	centroidOfCentroids := centroidSum.Mul(1.0 / float64(len(cloudCentroids)))

	// Find the cloud closest to the centroid-of-centroids
	closestCloudIndex := -1
	minDistanceToCentroid := math.Inf(1)
	for i, centroid := range cloudCentroids {
		distance := centroid.Distance(centroidOfCentroids)
		if distance < minDistanceToCentroid {
			minDistanceToCentroid = distance
			closestCloudIndex = i
		}
	}

	if closestCloudIndex == -1 {
		return nil, fmt.Errorf("no valid starting cloud found")
	}

	// Find the cloud with the greatest overlap to the closest-to-centroid cloud
	var bestInitialEdge *overlapEdge
	maxOverlapWithClosest := -1
	for i := range graph.edges {
		edge := &graph.edges[i]
		var overlapPoints int

		switch {
		case edge.cloudIndex1 == closestCloudIndex:
			overlapPoints = edge.overlapPoints
		case edge.cloudIndex2 == closestCloudIndex:
			overlapPoints = edge.overlapPoints
		default:
			continue // This edge doesn't involve the closest-to-centroid cloud
		}

		if overlapPoints > maxOverlapWithClosest {
			maxOverlapWithClosest = overlapPoints
			bestInitialEdge = edge
		}
	}

	if bestInitialEdge == nil {
		return nil, fmt.Errorf("closest-to-centroid cloud has no valid connections")
	}

	// First merge: closest-to-centroid cloud with its best partner
	mergeOrder = append(mergeOrder, bestInitialEdge)

	merged[bestInitialEdge.cloudIndex1] = true
	merged[bestInitialEdge.cloudIndex2] = true

	// Iteratively add remaining clouds with greatest overlap to merged group
	for mergedCount := 2; mergedCount < graph.cloudCount; mergedCount++ {
		var bestNextEdge *overlapEdge
		var bestUnmergedIndex int
		maxOverlapWithMerged := -1

		// Find the unmerged cloud with the best connection to any merged cloud
		for i := range graph.edges {
			edge := &graph.edges[i]

			// Check if this edge connects a merged cloud to an unmerged one
			var unmergedIndex int
			var validConnection bool

			if merged[edge.cloudIndex1] && !merged[edge.cloudIndex2] {
				unmergedIndex = edge.cloudIndex2
				validConnection = true
			} else if merged[edge.cloudIndex2] && !merged[edge.cloudIndex1] {
				unmergedIndex = edge.cloudIndex1
				validConnection = true
			}

			if validConnection && edge.overlapPoints > maxOverlapWithMerged {
				maxOverlapWithMerged = edge.overlapPoints
				bestNextEdge = edge
				bestUnmergedIndex = unmergedIndex
			}
		}

		if bestNextEdge == nil {
			// No more connections found - disconnected components are an error
			return nil, errors.New("disconnected component detected, cannot merge")
		}

		// Add this merge to the sequence
		mergeOrder = append(mergeOrder, bestNextEdge)

		merged[bestUnmergedIndex] = true
	}

	return mergeOrder, nil
}

// buildOverlapGraph computes pairwise overlaps between all point clouds in parallel.
func buildOverlapGraph(ctx context.Context, clouds []pointcloud.PointCloud, config ICPConfig, logger logging.Logger) (*overlapGraph, error) {
	n := len(clouds)
	if n < 2 {
		return nil, fmt.Errorf("need at least 2 point clouds, got %d", n)
	}

	// Build octrees for all clouds
	octrees := make([]*pointcloud.BasicOctree, 0, n)

	for _, cloud := range clouds {
		octree, err := pointcloud.ToBasicOctree(cloud, 0)
		if err != nil {
			return nil, err
		}
		octrees = append(octrees, octree)
	}

	// Generate all pairs for overlap computation
	type pairJob struct {
		i, j int
	}

	pairJobs := make([]pairJob, 0, n*(n-1)/2)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			pairJobs = append(pairJobs, pairJob{i, j})
		}
	}
	logger.Debugf("  Computing overlap for %d pointcloud pairs using %d workers", len(pairJobs), maxMultiAlignmentWorkers)

	// We'll compute overlaps in parallel, and collect the results in this channel.
	resultChan := make(chan overlapEdge, len(pairJobs))
	numWorkers := min(maxMultiAlignmentWorkers, len(pairJobs))
	workers := pcutils.NewParallelWorkers(numWorkers)

	// Error handling
	var mu sync.Mutex
	var firstError error

	for _, job := range pairJobs {
		workers.Do(func() {
			select {
			case <-ctx.Done():
				return
			default: // Time to actually do work
			}
			if firstError != nil {
				return
			}

			overlap, err := findOverlapRegion(octrees[job.i], octrees[job.j], config)
			if err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = err
				}
				mu.Unlock()
				return
			}

			edge := overlapEdge{
				cloudIndex1:   job.i,
				cloudIndex2:   job.j,
				overlapPoints: len(overlap.query),
				overlapRegion: overlap,
			}

			resultChan <- edge
		})
	}

	workers.Wait()
	close(resultChan)

	// Check if any errors occurred during processing
	if firstError != nil {
		return nil, firstError
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Collect results
	var edges []overlapEdge
	edgeMap := make(map[string]*overlapEdge)

	resultCount := 0
	for edge := range resultChan {
		resultCount++
		if edge.overlapPoints >= config.MinOverlapPoints {
			edges = append(edges, edge)
			key := fmt.Sprintf("%d,%d", edge.cloudIndex1, edge.cloudIndex2)
			edgeMap[key] = &edges[len(edges)-1]
			logger.Debugf("  Valid overlap: Cloud %d <-> Cloud %d (%d points)",
				edge.cloudIndex1, edge.cloudIndex2, edge.overlapPoints)
		} else {
			logger.Debugf("  Insufficient overlap: Cloud %d <-> Cloud %d (%d < %d points)",
				edge.cloudIndex1, edge.cloudIndex2, edge.overlapPoints, config.MinOverlapPoints)
		}
	}
	logger.Debugf("  Overlap computation complete: %d/%d pairs have sufficient overlap", len(edges), resultCount)

	return &overlapGraph{
		cloudCount: n,
		edges:      edges,
		edgeMap:    edgeMap,
	}, nil
}
