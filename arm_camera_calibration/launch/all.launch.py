from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="arm_camera_calibration",
                executable="calibration_gui",
                name="arm_camera_calibration_gui",
                output="screen",
                parameters=[
                    {
                        "image_topic": "/camera/image",
                        "pose_topic": "/arm/pose",
                        "pose_msg_type": "geometry_msgs/msg/PoseStamped",
                        "auto_start": False,
                        "chessboard_rows": 6,
                        "chessboard_cols": 9,
                        "square_size": 0.025,
                        "min_samples": 15,
                        "base_frame": "base_link",
                        "camera_frame": "camera_link",
                    }
                ],
            )
        ]
    )
