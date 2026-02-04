import importlib
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


@dataclass
class Sample:
    object_points: np.ndarray
    image_points: np.ndarray
    gripper_to_base: Tuple[np.ndarray, np.ndarray]


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - optional dependency
    least_squares = None


class ArmCameraCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_camera_calibration")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("pose_topic", "/arm/pose")
        self.declare_parameter("pose_msg_type", "geometry_msgs/msg/PoseStamped")
        self.declare_parameter("chessboard_rows", 6)
        self.declare_parameter("chessboard_cols", 9)
        self.declare_parameter("square_size", 0.025)
        self.declare_parameter("min_samples", 15)
        self.declare_parameter("handeye_method", "tsai")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("publish_camera_info", True)
        self.declare_parameter("publish_extrinsic", True)
        self.declare_parameter("extrinsic_topic", "/camera/extrinsic")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("refine_with_lm", False)
        self.declare_parameter("refine_max_iters", 50)
        self.declare_parameter("refine_reproj_weight", 1.0)
        self.declare_parameter("refine_motion_weight", 1.0)
        self.declare_parameter("optimize_tcp_offset", False)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.pose_msg_type_name = (
            self.get_parameter("pose_msg_type").get_parameter_value().string_value
        )
        self.chessboard_rows = (
            self.get_parameter("chessboard_rows").get_parameter_value().integer_value
        )
        self.chessboard_cols = (
            self.get_parameter("chessboard_cols").get_parameter_value().integer_value
        )
        self.square_size = (
            self.get_parameter("square_size").get_parameter_value().double_value
        )
        self.min_samples = (
            self.get_parameter("min_samples").get_parameter_value().integer_value
        )
        self.handeye_method = (
            self.get_parameter("handeye_method").get_parameter_value().string_value
        )
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.publish_camera_info = (
            self.get_parameter("publish_camera_info").get_parameter_value().bool_value
        )
        self.publish_extrinsic = (
            self.get_parameter("publish_extrinsic").get_parameter_value().bool_value
        )
        self.extrinsic_topic = (
            self.get_parameter("extrinsic_topic").get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.refine_with_lm = (
            self.get_parameter("refine_with_lm").get_parameter_value().bool_value
        )
        self.refine_max_iters = (
            self.get_parameter("refine_max_iters").get_parameter_value().integer_value
        )
        self.refine_reproj_weight = (
            self.get_parameter("refine_reproj_weight").get_parameter_value().double_value
        )
        self.refine_motion_weight = (
            self.get_parameter("refine_motion_weight").get_parameter_value().double_value
        )
        self.optimize_tcp_offset = (
            self.get_parameter("optimize_tcp_offset").get_parameter_value().bool_value
        )

        self.samples: List[Sample] = []
        self.bridge = CvBridge()
        self.image_size: Optional[Tuple[int, int]] = None

        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.extrinsic_pub = self.create_publisher(
            TransformStamped, self.extrinsic_topic, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        image_sub = message_filters.Subscriber(self, Image, image_topic)
        pose_msg_type = self._load_pose_msg_type(self.pose_msg_type_name)
        pose_sub = message_filters.Subscriber(self, pose_msg_type, pose_topic)
        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, pose_sub], queue_size=10, slop=0.1
        )
        sync.registerCallback(self._sync_callback)

        self.calibrate_srv = self.create_service(Trigger, "calibrate", self._on_calibrate)
        self.clear_srv = self.create_service(Trigger, "clear_samples", self._on_clear)

        self.get_logger().info(
            "Arm-camera calibration node started. Waiting for synced image and pose data."
        )

    def _sync_callback(self, image_msg: Image, pose_msg: object) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (self.chessboard_cols, self.chessboard_rows)
        )
        if not found:
            self.get_logger().debug("Chessboard not found in current image.")
            return

        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objp = self._create_object_points()
        r_gripper2base, t_gripper2base = self._pose_to_gripper2base(pose_msg)
        sample = Sample(
            object_points=objp,
            image_points=corners,
            gripper_to_base=(r_gripper2base, t_gripper2base),
        )
        self.samples.append(sample)
        self.image_size = (gray.shape[1], gray.shape[0])
        self.get_logger().info(
            f"Captured sample {len(self.samples)} with chessboard detection."
        )

    def _on_clear(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.samples.clear()
        self.image_size = None
        response.success = True
        response.message = "Cleared calibration samples."
        self.get_logger().info(response.message)
        return response

    def _on_calibrate(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if len(self.samples) < self.min_samples:
            response.success = False
            response.message = (
                f"Need at least {self.min_samples} samples, got {len(self.samples)}."
            )
            return response
        if self.image_size is None:
            response.success = False
            response.message = "No valid image size recorded."
            return response

        objpoints = [sample.object_points for sample in self.samples]
        imgpoints = [sample.image_points for sample in self.samples]

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        if not ret:
            response.success = False
            response.message = "Camera calibration failed."
            return response

        handeye_method = HAND_EYE_METHODS.get(self.handeye_method.lower())
        if handeye_method is None:
            response.success = False
            response.message = f"Unknown hand-eye method: {self.handeye_method}."
            return response

        r_gripper2base, t_gripper2base = zip(
            *[sample.gripper_to_base for sample in self.samples]
        )
        r_target2cam = []
        t_target2cam = []
        for rvec, tvec in zip(rvecs, tvecs):
            rmat, _ = cv2.Rodrigues(rvec)
            r_target2cam.append(rmat)
            t_target2cam.append(tvec.reshape(3, 1))

        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=handeye_method,
        )

        if self.refine_with_lm:
            refined = self._refine_solution(
                camera_matrix,
                dist_coeffs,
                r_cam2gripper,
                t_cam2gripper,
                r_gripper2base,
                t_gripper2base,
                objpoints,
                imgpoints,
            )
            if refined is not None:
                camera_matrix, dist_coeffs, r_cam2gripper, t_cam2gripper = refined

        r_base2gripper, t_base2gripper = self._invert_transform(
            r_gripper2base[0], t_gripper2base[0]
        )
        r_gripper2cam, t_gripper2cam = self._invert_transform(r_cam2gripper, t_cam2gripper)
        r_base2cam, t_base2cam = self._compose_transform(
            r_base2gripper, t_base2gripper, r_gripper2cam, t_gripper2cam
        )

        if self.publish_camera_info:
            camera_info = self._build_camera_info(camera_matrix, dist_coeffs)
            self.camera_info_pub.publish(camera_info)

        if self.publish_extrinsic:
            transform_msg = self._build_extrinsic_msg(r_base2cam, t_base2cam)
            self.extrinsic_pub.publish(transform_msg)
            self.tf_broadcaster.sendTransform(transform_msg)

        response.success = True
        response.message = "Calibration completed and results published."
        self.get_logger().info(response.message)
        return response

    def _build_camera_info(self, camera_matrix: np.ndarray, dist: np.ndarray) -> CameraInfo:
        camera_info = CameraInfo()
        camera_info.width = self.image_size[0]
        camera_info.height = self.image_size[1]
        camera_info.k = camera_matrix.flatten().tolist()
        camera_info.d = dist.flatten().tolist()
        camera_info.r = np.eye(3).flatten().tolist()
        camera_info.p = (
            np.array(
                [
                    [camera_matrix[0, 0], 0.0, camera_matrix[0, 2], 0.0],
                    [0.0, camera_matrix[1, 1], camera_matrix[1, 2], 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            .flatten()
            .tolist()
        )
        camera_info.header.frame_id = self.camera_frame
        return camera_info

    def _build_extrinsic_msg(
        self, rotation: np.ndarray, translation: np.ndarray
    ) -> TransformStamped:
        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = self.base_frame
        transform_msg.child_frame_id = self.camera_frame
        quat = self._rotation_to_quaternion(rotation)
        transform_msg.transform.translation.x = float(translation[0])
        transform_msg.transform.translation.y = float(translation[1])
        transform_msg.transform.translation.z = float(translation[2])
        transform_msg.transform.rotation.x = quat[0]
        transform_msg.transform.rotation.y = quat[1]
        transform_msg.transform.rotation.z = quat[2]
        transform_msg.transform.rotation.w = quat[3]
        return transform_msg

    def _refine_solution(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        r_cam2gripper: np.ndarray,
        t_cam2gripper: np.ndarray,
        r_gripper2base: Tuple[np.ndarray, ...],
        t_gripper2base: Tuple[np.ndarray, ...],
        objpoints: List[np.ndarray],
        imgpoints: List[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if least_squares is None:
            self.get_logger().warning(
                "SciPy is not available; skipping LM refinement."
            )
            return None

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        dist = dist_coeffs.flatten()
        dist = np.pad(dist, (0, max(0, 5 - dist.size)), mode="constant")[:5]

        rvec_cam2gripper, _ = cv2.Rodrigues(r_cam2gripper)
        tvec_cam2gripper = t_cam2gripper.reshape(3)

        tcp_rvec = np.zeros(3)
        tcp_tvec = np.zeros(3)

        params = [fx, fy, cx, cy, *dist, *rvec_cam2gripper.reshape(3), *tvec_cam2gripper]
        if self.optimize_tcp_offset:
            params += [*tcp_rvec, *tcp_tvec]

        params = np.array(params, dtype=float)
        weight_reproj = float(self.refine_reproj_weight)
        weight_motion = float(self.refine_motion_weight)

        def residuals(values: np.ndarray) -> np.ndarray:
            idx = 0
            fx_v, fy_v, cx_v, cy_v = values[idx : idx + 4]
            idx += 4
            dist_v = values[idx : idx + 5]
            idx += 5
            rvec_x = values[idx : idx + 3]
            idx += 3
            tvec_x = values[idx : idx + 3]
            idx += 3
            if self.optimize_tcp_offset:
                tcp_r = values[idx : idx + 3]
                tcp_t = values[idx + 3 : idx + 6]
            else:
                tcp_r = tcp_rvec
                tcp_t = tcp_tvec

            cam_matrix = np.array(
                [[fx_v, 0.0, cx_v], [0.0, fy_v, cy_v], [0.0, 0.0, 1.0]],
                dtype=float,
            )
            dist_vec = dist_v.reshape(-1, 1)

            rmat_x, _ = cv2.Rodrigues(rvec_x)
            tvec_x_mat = tvec_x.reshape(3, 1)
            residual_list: List[np.ndarray] = []

            target_poses = []
            for objp, imgp in zip(objpoints, imgpoints):
                ok, rvec_t, tvec_t = cv2.solvePnP(
                    objp, imgp, cam_matrix, dist_vec, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    return np.full(1, 1e6)
                target_poses.append((rvec_t, tvec_t))
                proj, _ = cv2.projectPoints(objp, rvec_t, tvec_t, cam_matrix, dist_vec)
                reproj = (proj.reshape(-1, 2) - imgp.reshape(-1, 2)).reshape(-1)
                residual_list.append(weight_reproj * reproj)

            if len(target_poses) >= 2:
                tcp_rmat, _ = cv2.Rodrigues(tcp_r)
                tcp_tvec = tcp_t.reshape(3, 1)
                r_tcp, t_tcp = tcp_rmat, tcp_tvec

                base_to_tcp = []
                for r_g2b, t_g2b in zip(r_gripper2base, t_gripper2base):
                    r_b2g, t_b2g = self._invert_transform(r_g2b, t_g2b)
                    r_b2tcp, t_b2tcp = self._compose_transform(r_b2g, t_b2g, r_tcp, t_tcp)
                    base_to_tcp.append((r_b2tcp, t_b2tcp))

                for idx_pair in range(len(target_poses) - 1):
                    rvec_ti, tvec_ti = target_poses[idx_pair]
                    rvec_tj, tvec_tj = target_poses[idx_pair + 1]
                    r_ti, _ = cv2.Rodrigues(rvec_ti)
                    r_tj, _ = cv2.Rodrigues(rvec_tj)
                    t_ti = tvec_ti.reshape(3, 1)
                    t_tj = tvec_tj.reshape(3, 1)
                    r_ti_inv, t_ti_inv = self._invert_transform(r_ti, t_ti)
                    r_b, t_b = self._compose_transform(r_ti_inv, t_ti_inv, r_tj, t_tj)

                    r_b2tcp_i, t_b2tcp_i = base_to_tcp[idx_pair]
                    r_b2tcp_j, t_b2tcp_j = base_to_tcp[idx_pair + 1]
                    r_tcp_i_inv, t_tcp_i_inv = self._invert_transform(r_b2tcp_i, t_b2tcp_i)
                    r_a, t_a = self._compose_transform(
                        r_tcp_i_inv, t_tcp_i_inv, r_b2tcp_j, t_b2tcp_j
                    )

                    r_left, t_left = self._compose_transform(
                        r_a, t_a, rmat_x, tvec_x_mat
                    )
                    r_right, t_right = self._compose_transform(
                        rmat_x, tvec_x_mat, r_b, t_b
                    )
                    r_delta, t_delta = self._compose_transform(
                        *self._invert_transform(r_left, t_left), r_right, t_right
                    )
                    rvec_delta, _ = cv2.Rodrigues(r_delta)
                    motion_res = np.hstack([rvec_delta.reshape(3), t_delta.reshape(3)])
                    residual_list.append(weight_motion * motion_res)

            if not residual_list:
                return np.zeros(1)

            return np.concatenate(residual_list)

        res = residuals(params)
        method = "lm" if res.size >= params.size else "trf"
        if method != "lm":
            self.get_logger().warning(
                "LM refinement requires at least as many residuals as parameters; "
                "falling back to trust region."
            )
        result = least_squares(residuals, params, method=method, max_nfev=self.refine_max_iters)

        if not result.success:
            self.get_logger().warning(
                f"LM refinement did not converge: {result.message}"
            )
            return None

        refined = result.x
        fx, fy, cx, cy = refined[:4]
        dist = refined[4:9]
        idx = 9
        rvec_x = refined[idx : idx + 3]
        idx += 3
        tvec_x = refined[idx : idx + 3]

        cam_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float
        )
        dist_vec = dist.reshape(-1, 1)
        rmat_x, _ = cv2.Rodrigues(rvec_x)
        tvec_x = tvec_x.reshape(3, 1)
        self.get_logger().info("LM refinement succeeded.")
        return cam_matrix, dist_vec, rmat_x, tvec_x

    def _create_object_points(self) -> np.ndarray:
        grid = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0 : self.chessboard_cols, 0 : self.chessboard_rows].T.reshape(
            -1, 2
        )
        grid *= self.square_size
        return grid

    def _pose_to_gripper2base(self, pose_msg: object) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(pose_msg, "pose"):
            position = pose_msg.pose.position
            orientation = pose_msg.pose.orientation
            r_base2gripper = self._quaternion_to_rotation(
                np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            )
            t_base2gripper = np.array([position.x, position.y, position.z]).reshape(3, 1)
        elif all(
            hasattr(pose_msg, attr)
            for attr in ("x_pos", "y_pos", "z_pos", "rx_pos", "ry_pos", "rz_pos")
        ):
            t_base2gripper = np.array(
                [
                    pose_msg.x_pos.data,
                    pose_msg.y_pos.data,
                    pose_msg.z_pos.data,
                ]
            ).reshape(3, 1)
            r_base2gripper = self._rpy_to_rotation(
                pose_msg.rx_pos.data,
                pose_msg.ry_pos.data,
                pose_msg.rz_pos.data,
            )
        else:
            raise ValueError(
                "Unsupported pose message type. Expected PoseStamped or EndPosStruct-like."
            )
        r_gripper2base, t_gripper2base = self._invert_transform(
            r_base2gripper, t_base2gripper
        )
        return r_gripper2base, t_gripper2base

    @staticmethod
    def _invert_transform(
        rotation: np.ndarray, translation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r_inv = rotation.T
        t_inv = -r_inv @ translation
        return r_inv, t_inv

    @staticmethod
    def _compose_transform(
        r_a: np.ndarray, t_a: np.ndarray, r_b: np.ndarray, t_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r = r_a @ r_b
        t = r_a @ t_b + t_a
        return r, t

    @staticmethod
    def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    @staticmethod
    def _load_pose_msg_type(type_name: str) -> type:
        try:
            module_name, class_name = type_name.split("/")
            module = importlib.import_module(f"{module_name}.msg")
            return getattr(module, class_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import pose_msg_type '{type_name}': {exc}"
            ) from exc

    @staticmethod
    def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = quaternion
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            raise ValueError("Quaternion has zero norm.")
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
        trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rotation[2, 1] - rotation[1, 2]) / s
            y = (rotation[0, 2] - rotation[2, 0]) / s
            z = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
        return (float(x), float(y), float(z), float(w))


def main() -> None:
    rclpy.init()
    node = ArmCameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
