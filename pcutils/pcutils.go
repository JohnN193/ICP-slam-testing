// Package pcutils provides point cloud utility functions used by the ICP alignment implementation.
package pcutils

import (
	"fmt"
	"math"
	"sort"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/pointcloud"
	"gonum.org/v1/gonum/mat"
)

// VoxelKey identifies a voxel in 3D space.
type VoxelKey struct {
	X, Y, Z int
}

// VoxelData holds the accumulated data for a single voxel.
type VoxelData struct {
	Centroid      r3.Vector
	Count         int
	OriginalIndex int
}

// VoxelDownsample performs voxel-based downsampling to maintain spatial distribution.
func VoxelDownsample(points []r3.Vector, voxelSize float64) []r3.Vector {
	if len(points) == 0 || voxelSize == 0 {
		return points
	}

	// Find bounding box for coordinate transformation
	minBounds, maxBounds := points[0], points[0]
	for _, p := range points[1:] {
		if p.X < minBounds.X {
			minBounds.X = p.X
		}
		if p.Y < minBounds.Y {
			minBounds.Y = p.Y
		}
		if p.Z < minBounds.Z {
			minBounds.Z = p.Z
		}
		if p.X > maxBounds.X {
			maxBounds.X = p.X
		}
		if p.Y > maxBounds.Y {
			maxBounds.Y = p.Y
		}
		if p.Z > maxBounds.Z {
			maxBounds.Z = p.Z
		}
	}

	voxelMap := make(map[VoxelKey]VoxelData)

	for i, point := range points {
		key := VoxelKey{
			X: int(math.Floor((point.X - minBounds.X) / voxelSize)),
			Y: int(math.Floor((point.Y - minBounds.Y) / voxelSize)),
			Z: int(math.Floor((point.Z - minBounds.Z) / voxelSize)),
		}

		if voxel, exists := voxelMap[key]; exists {
			voxel.Count++
			voxel.Centroid = voxel.Centroid.Add(point)
			voxelMap[key] = voxel
		} else {
			voxelMap[key] = VoxelData{
				Centroid:      point,
				Count:         1,
				OriginalIndex: i,
			}
		}
	}

	// Extract representative points (centroids) in deterministic order
	keys := make([]VoxelKey, 0, len(voxelMap))
	for key := range voxelMap {
		keys = append(keys, key)
	}

	// Sort keys for deterministic order (X, then Y, then Z)
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].X != keys[j].X {
			return keys[i].X < keys[j].X
		}
		if keys[i].Y != keys[j].Y {
			return keys[i].Y < keys[j].Y
		}
		return keys[i].Z < keys[j].Z
	})

	sampled := make([]r3.Vector, 0, len(voxelMap))
	for _, key := range keys {
		voxel := voxelMap[key]
		centroid := voxel.Centroid.Mul(1.0 / float64(voxel.Count))
		sampled = append(sampled, centroid)
	}

	return sampled
}

// VectorsToOctree converts a slice of r3.Vector to a BasicOctree.
func VectorsToOctree(points []r3.Vector) (*pointcloud.BasicOctree, error) {
	pc := pointcloud.NewBasicEmpty()
	for _, point := range points {
		err := pc.Set(point, pointcloud.NewBasicData())
		if err != nil {
			return nil, err
		}
	}
	return pointcloud.ToBasicOctree(pc, 0)
}

// ComputeCentroid calculates the centroid of a set of points.
func ComputeCentroid(points []r3.Vector) r3.Vector {
	sum := r3.Vector{X: 0, Y: 0, Z: 0}
	for _, p := range points {
		sum = sum.Add(p)
	}
	return sum.Mul(1.0 / float64(len(points)))
}

// ComputePrincipalDirections computes the principal directions using PCA.
func ComputePrincipalDirections(points []r3.Vector, center r3.Vector) (r3.Vector, r3.Vector, error) {
	n := len(points)
	if n < 3 {
		return r3.Vector{}, r3.Vector{}, fmt.Errorf("need at least 3 points for PCA")
	}

	// Create centered data matrix
	pcaPoints := make([]r3.Vector, 0, n/2)
	for i := 0; i < n; i++ {
		pcaPoints = append(pcaPoints, points[i].Sub(center))
	}

	// Build covariance matrix manually for better performance
	var cxx, cxy, cxz, cyy, cyz, czz float64
	np := len(pcaPoints)

	for _, p := range pcaPoints {
		cxx += p.X * p.X
		cxy += p.X * p.Y
		cxz += p.X * p.Z
		cyy += p.Y * p.Y
		cyz += p.Y * p.Z
		czz += p.Z * p.Z
	}

	// Normalize
	factor := 1.0 / float64(np-1)
	cxx *= factor
	cxy *= factor
	cxz *= factor
	cyy *= factor
	cyz *= factor
	czz *= factor

	// Create covariance matrix
	covData := []float64{
		cxx, cxy, cxz,
		cxy, cyy, cyz,
		cxz, cyz, czz,
	}
	cov := mat.NewDense(3, 3, covData)

	// Perform eigendecomposition
	var eigen mat.Eigen
	ok := eigen.Factorize(cov, mat.EigenRight)
	if !ok {
		return r3.Vector{}, r3.Vector{}, fmt.Errorf("eigendecomposition failed")
	}

	// Get eigenvalues and eigenvectors
	eigenVals := eigen.Values(nil)
	eigenVecs := &mat.CDense{}
	eigen.VectorsTo(eigenVecs)

	// Find indices of largest and second largest eigenvalues
	var maxIdx, secondIdx int
	var maxVal, secondVal float64

	for i := 0; i < 3; i++ {
		val := real(eigenVals[i])
		if val > maxVal {
			secondVal, secondIdx = maxVal, maxIdx
			maxVal, maxIdx = val, i
		} else if val > secondVal {
			secondVal, secondIdx = val, i
		}
	}

	// Extract principal directions
	principalDir := r3.Vector{
		X: real(eigenVecs.At(0, maxIdx)),
		Y: real(eigenVecs.At(1, maxIdx)),
		Z: real(eigenVecs.At(2, maxIdx)),
	}

	secondaryDir := r3.Vector{
		X: real(eigenVecs.At(0, secondIdx)),
		Y: real(eigenVecs.At(1, secondIdx)),
		Z: real(eigenVecs.At(2, secondIdx)),
	}

	return principalDir.Normalize(), secondaryDir.Normalize(), nil
}

// MeasurePlaneFitness measures how well a pointcloud fits a plane defined by a point and normal.
// Returns the root mean square (RMS) distance of all points from the plane.
func MeasurePlaneFitness(cloud pointcloud.PointCloud, planePoint, planeNormal r3.Vector) (float64, error) {
	if cloud.Size() == 0 {
		return 0, fmt.Errorf("cannot measure plane fitness of empty point cloud")
	}

	normalLength := planeNormal.Norm()
	if normalLength == 0 {
		return math.Inf(1), fmt.Errorf("plane normal vector cannot be zero")
	}
	unitNormal := planeNormal.Mul(1.0 / normalLength)

	octree, err := pointcloud.ToBasicOctree(cloud, 0)
	if err != nil {
		return math.Inf(1), fmt.Errorf("failed to convert point cloud to octree: %w", err)
	}

	points := octree.ToPoints(0)
	if len(points) == 0 {
		return math.Inf(1), fmt.Errorf("octree contains no points")
	}

	var sumSquaredDistances float64
	for _, point := range points {
		pointVector := point.Sub(planePoint)
		distance := pointVector.Dot(unitNormal)
		sumSquaredDistances += distance * distance
	}

	meanSquaredDistance := sumSquaredDistances / float64(len(points))
	return math.Sqrt(meanSquaredDistance), nil
}
