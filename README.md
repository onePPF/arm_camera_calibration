# arm_camera_calibration

ROS 2 package that performs joint camera intrinsic calibration and arm-camera extrinsic calibration
using synchronized robot poses and chessboard images.

## Features

- Subscribes to a 6D robot pose (`geometry_msgs/PoseStamped`) and a chessboard image
  (`sensor_msgs/Image`).
- Detects chessboard corners and accumulates samples.
- Computes camera intrinsics with OpenCV.
- Computes camera extrinsics relative to a robot base frame using hand-eye calibration.
- Publishes `sensor_msgs/CameraInfo` and a `geometry_msgs/TransformStamped` for the extrinsic.

## Usage

```bash
colcon build --packages-select arm_camera_calibration
. install/setup.bash
ros2 launch arm_camera_calibration calibration.launch.py
```

Launch the GUI (shows images, progress bars, and control buttons):

```bash
ros2 launch arm_camera_calibration gui.launch.py
```

Launch the full workflow with GUI from a single launch file:

```bash
ros2 launch arm_camera_calibration all.launch.py
```

Optional refinement uses SciPy's LM solver. Ensure `python3-scipy` is installed if you enable
`refine_with_lm`.

Trigger calibration after collecting samples:

```bash
ros2 service call /calibrate std_srvs/srv/Trigger {}
```

Clear samples if needed:

```bash
ros2 service call /clear_samples std_srvs/srv/Trigger {}
```

## Key Parameters

- `image_topic`: Image topic (default: `/camera/image`).
- `pose_topic`: Pose topic (default: `/arm/pose`).
- `pose_msg_type`: Pose message type (default: `geometry_msgs/msg/PoseStamped`). For
  `EndPosStruct`, set to `<your_package>/msg/EndPosStruct`; `rx_pos/ry_pos/rz_pos` are
  interpreted as roll/pitch/yaw in radians.
- `auto_start`: Start collecting immediately (default: `true`). Set `false` for GUI workflows.
- `coverage_bins`: Number of bins per coverage metric (default: `5`).
- `x_min` / `x_max`: Coverage range for X in meters (default: `-0.1` to `0.1`).
- `y_min` / `y_max`: Coverage range for Y in meters (default: `-0.1` to `0.1`).
- `size_min` / `size_max`: Coverage range for board size ratio (default: `0.02` to `0.2`).
- `skew_min` / `skew_max`: Coverage range for skew angle in degrees (default: `0` to `20`).
- `chessboard_rows`: Inner corner rows (default: `6`).
- `chessboard_cols`: Inner corner cols (default: `9`).
- `square_size`: Chessboard square size in meters (default: `0.025`).
- `min_samples`: Minimum samples required (default: `15`).
- `handeye_method`: Hand-eye algorithm (`tsai`, `park`, `horaud`, `andreff`, `daniilidis`).
- `base_frame`: Output base frame (default: `base_link`).
- `camera_frame`: Output camera frame (default: `camera_link`).
- `camera_info_topic`: Output camera info topic (default: `/camera/camera_info`).
- `extrinsic_topic`: Output extrinsic topic (default: `/camera/extrinsic`).
- `refine_with_lm`: Enable LM refinement using SciPy (default: `false`).
- `refine_max_iters`: Maximum LM iterations (default: `50`).
- `refine_reproj_weight`: Weight for reprojection residuals (default: `1.0`).
- `refine_motion_weight`: Weight for AX=XB motion residuals (default: `1.0`).
- `optimize_tcp_offset`: Optimize TCP offset during refinement (default: `false`).
