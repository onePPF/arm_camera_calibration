from setuptools import setup

package_name = "arm_camera_calibration"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/calibration.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="ROS 2 tool for joint camera intrinsic and arm-camera extrinsic calibration.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "calibration_node = arm_camera_calibration.calibration_node:main",
        ],
    },
)
