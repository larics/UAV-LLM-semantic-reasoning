#!/usr/bin/env python3
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import yaml
import numpy as np
import cv2
import struct


def load_camera_params(yaml_path):
    """Load camera intrinsics and distortion from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    camera = config['camera']
    K = np.array(camera['camera_matrix']['data'],  dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.array(camera['distortion_coefficients']['data'], dtype=np.float32)
    image_width = camera['image_width']
    image_height = camera['image_height']
    return K, dist_coeffs, image_width, image_height

# Kamera na pravoj letjelici
# def pixel_to_3d_direction(u, v, K, dist_coeffs=None):
#     """
#     Convert pixel (u,v) to a unit 3D direction vector in the camera coordinate system.
#     """
#     # If you have OpenCV:
#     pts = np.array([[u, v]], dtype=np.float32)[None, ...]  # shape (1,1,2)
#     if dist_coeffs is not None and dist_coeffs.size > 0:
#         undist_pts = cv2.undistortPoints(pts, K, dist_coeffs)  # shape (1,1,2)
#         x_undist = -undist_pts[0,0,0]
#         y_undist = undist_pts[0,0,1]

#     d_cam = np.array([x_undist, y_undist, 1.0], dtype=np.float64)
#     d_cam /= np.linalg.norm(d_cam)
#     return d_cam

# Kamera u simulaciji
import math
HFOV = 1.5184
WIDTH = 324
HEIGHT = 324
f_x = (WIDTH / 2.0) / math.tan(HFOV / 2.0)
f_y = f_x  # square image
cx = WIDTH / 2.0
cy = HEIGHT / 2.0

def pixel_to_3d_direction(u, v, K, dist_coeffs=None) -> np.ndarray:
    x = (u - cx) / f_x
    y = (v - cy) / f_y
    z = 1.0

    direction = np.array([-x, y, z], dtype=float)
    direction /= np.linalg.norm(direction)
    return direction

def camera_to_world_rotation(roll, pitch, yaw):
    """
    Returns rotation matrix from the camera frame to the world frame,
    incorporating roll, pitch, and yaw.
    """
    # Camera-to-drone rotation (fixed)
    R_cam_drone = np.array([
        [0,  0,  1],                # Camera +Z -> Drone +X
        [1,  0,  0],                # Camera +X -> Drone +Y
        [0, -1,  0]                 # Camera +Y -> Drone -Z
    ], dtype=np.float32)

    # Rotation about X-axis (roll)
    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)]
    ], dtype=np.float32)

    # Rotation about Y-axis (pitch)
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ], dtype=np.float32)

    # Rotation about Z-axis (yaw)
    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [ 0,            0,           1]
    ], dtype=np.float32)

    # Drone-to-world rotation (apply roll, then pitch, then yaw in that order)
    R_drone_world = Rz @ Ry @ Rx

    # Final rotation: world <- drone <- camera
    R_world_camera = R_drone_world @ R_cam_drone
    return R_world_camera


def xyz_array_to_pointcloud2(cf, points_xyz, frame_id="world"):
    """
    Convert an array into a PointCloud2 message.
    """
    pc = PointCloud2()
    pc.header = Header()
    pc.header.frame_id = frame_id
    pc.header.stamp = cf.get_clock().now().to_msg()
    pc.height = 1
    pc.width = points_xyz.shape[0]
    pc.is_bigendian = False
    pc.point_step = 12  # 3 floats * 4 bytes
    pc.row_step = pc.point_step * points_xyz.shape[0]
    pc.is_dense = True

    # Fields: x, y, z
    pc.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1)
    ]

    # Create the binary data
    if points_xyz.shape[0] == 0:
        pc.data = b''  # Empty message if no points
    else:
        data_buffer = [struct.pack('<fff', x, y, z) for x, y, z in points_xyz]
        pc.data = b''.join(data_buffer)

    return pc


if __name__ == '__main__':
    main()
