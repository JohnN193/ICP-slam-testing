# ICP SLAM

A Viam module that builds a 3D point cloud map by accumulating scans from a depth camera and aligning them using Iterative Closest Point (ICP).

## How it works

1. Call the `add_scan` DoCommand to pull a point cloud from the configured camera
2. The scan is transformed into the world frame using Viam's frame system
3. ICP aligns the new scan against all previously accumulated scans
4. The merged map is available via the standard SLAM `PointCloudMap` API

## Configure your `icp-slam` service

Add the following to your robot config:

```json
{
  "name": "my-icp-slam",
  "api": "rdk:service:slam",
  "model": "cjnj193:icp:icp-slam",
  "attributes": {
    "camera": "<camera-name>",
    "movement_sensor": "<movement-sensor-name>"
  }
}
```

### Attributes

| Name              | Type   | Required | Description                                                     |
|-------------------|--------|----------|-----------------------------------------------------------------|
| `camera`          | string | Yes      | Name of the depth camera component to pull point clouds from    |
| `movement_sensor` | string | No       | Name of a movement sensor / odometry source (future use)        |

## DoCommands

### `add_scan`

Pulls the current point cloud from the camera, transforms it to world frame, and runs ICP alignment against all accumulated scans.

```json
{ "add_scan": true }
```

**Response:**

```json
{
  "scans_accumulated": 3,
  "map_points": 42891
}
```

## API

This module implements the standard Viam [SLAM service API](https://docs.viam.com/dev/reference/apis/services/slam/):

| Method           | Behaviour                                              |
|------------------|--------------------------------------------------------|
| `PointCloudMap`  | Returns the current ICP-aligned map as a PCD binary    |
| `Position`       | Returns a zero pose (localization not yet implemented) |
| `InternalState`  | Returns empty                                          |
| `Properties`     | Reports `MappingModeNewMap`, lists the camera sensor   |

## Notes

- Point clouds are expected in **meters** (PCD standard); ICP runs internally in **millimeters**
- At least two scans are required before alignment begins
- ICP uses NLOpt for optimization and requires CGo at build time
