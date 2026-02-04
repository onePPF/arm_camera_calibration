import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Optional

import cv2
import numpy as np
import rclpy
import rclpy.parameter
from std_srvs.srv import Trigger

from arm_camera_calibration.calibration_node import ArmCameraCalibrationNode


class ArmCameraCalibrationGuiNode(ArmCameraCalibrationNode):
    def __init__(self) -> None:
        super().__init__()
        self.set_parameters([
            rclpy.parameter.Parameter("auto_start", rclpy.Parameter.Type.BOOL, False)
        ])
        self.restart_collection()

    def calibrate_now(self) -> Trigger.Response:
        response = Trigger.Response()
        response = self._on_calibrate(Trigger.Request(), response)
        return response


class CalibrationGui:
    def __init__(self, node: ArmCameraCalibrationGuiNode) -> None:
        self.node = node
        self.root = tk.Tk()
        self.root.title("Arm-Camera Calibration")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

        self.start_button = tk.Button(self.root, text="Start", command=self._on_start)
        self.restart_button = tk.Button(
            self.root, text="Restart", command=self._on_restart
        )
        self.calib_button = tk.Button(self.root, text="Calib", command=self._on_calib)
        self.save_button = tk.Button(self.root, text="Save", command=self._on_save)

        self.start_button.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        self.restart_button.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        self.calib_button.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        self.save_button.grid(row=3, column=1, sticky="ew", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.grid(row=4, column=1, sticky="ew", padx=10, pady=5)

        self.progress_bars = {}
        self._init_progress("x", 5)
        self._init_progress("y", 6)
        self._init_progress("size", 7)
        self._init_progress("skew", 8)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

        self.photo_image: Optional[tk.PhotoImage] = None
        self.running = True

        self._update_gui()

    def _init_progress(self, label: str, row: int) -> None:
        ttk.Label(self.root, text=f"{label.upper()} Coverage").grid(
            row=row, column=0, sticky="w", padx=10
        )
        bar = ttk.Progressbar(self.root, orient="horizontal", length=200, mode="determinate")
        bar.grid(row=row, column=1, sticky="ew", padx=10, pady=2)
        self.progress_bars[label] = bar

    def _update_gui(self) -> None:
        image = self.node.get_display_image()
        if image is not None:
            self._update_image(image)
        self._update_progress()
        if self.running:
            self.root.after(200, self._update_gui)

    def _update_image(self, image: np.ndarray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        max_width = 640
        scale = min(1.0, max_width / float(width))
        if scale != 1.0:
            rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)))
            height, width = rgb.shape[:2]
        ppm_header = f"P6 {width} {height} 255\n"
        ppm_data = ppm_header.encode("ascii") + rgb.tobytes()
        self.photo_image = tk.PhotoImage(data=ppm_data)
        self.image_label.configure(image=self.photo_image)

    def _update_progress(self) -> None:
        progress = self.node.get_coverage_progress()
        for key, bar in self.progress_bars.items():
            value = int(progress.get(key, 0.0) * 100)
            bar["value"] = value
            self._apply_progress_color(bar, value)

    @staticmethod
    def _apply_progress_color(bar: ttk.Progressbar, value: int) -> None:
        style = ttk.Style()
        if value >= 100:
            color = "#4caf50"
        elif value >= 50:
            color = "#ffb300"
        else:
            color = "#e53935"
        style_name = f"{id(bar)}.Horizontal.TProgressbar"
        style.configure(style_name, troughcolor="#eeeeee", background=color)
        bar.configure(style=style_name)

    def _on_start(self) -> None:
        success, message = self.node.start_collection()
        self.status_var.set(message)

    def _on_restart(self) -> None:
        self.node.restart_collection()
        self.status_var.set("Restarted")

    def _on_calib(self) -> None:
        response = self.node.calibrate_now()
        self.status_var.set(response.message)

    def _on_save(self) -> None:
        if not self.node.last_calibration:
            self.status_var.set("No calibration results to save.")
            return
        directory = filedialog.askdirectory()
        if not directory:
            return
        path = Path(directory) / f"calibration_{int(time.time())}.json"
        payload = {
            "camera_matrix": self.node.last_calibration["camera_matrix"].tolist(),
            "dist_coeffs": self.node.last_calibration["dist_coeffs"].tolist(),
            "r_base2cam": self.node.last_calibration["r_base2cam"].tolist(),
            "t_base2cam": self.node.last_calibration["t_base2cam"].tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_var.set(f"Saved to {path}")

    def _on_close(self) -> None:
        self.running = False
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    rclpy.init()
    node = ArmCameraCalibrationGuiNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    gui = CalibrationGui(node)
    try:
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
