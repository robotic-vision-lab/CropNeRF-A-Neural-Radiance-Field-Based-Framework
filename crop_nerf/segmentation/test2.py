import numpy as np
import open3d as o3d
import os
from math import radians

def get_intrinsic(width, height):
    # return np.asarray([[935,     0,     width * 0.5],
    #                     [0,      935,   height * 0.5],
    #                         [0, 0, 1]])
    # 935.3074360871938,0.0,961.5
    # 0.0,935.3074360871938,539.5
    # 0.0,0.0,1.0
    cx = 961.93
    cy = 723.24
    fx =1442.47
    fy = 1442.47
    h = 1440
    w = 1920
    return np.asarray([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])

def get_extrinsic(K):
    camera_to_worlds = np.asarray([
        [0.046825625,0.9985399,0.026933009,0.047231793],
[-0.67224306,0.051444445,-0.73854095,-0.931187],
[-0.73884815,0.016477115,0.6736705,0.2662813],
[0,0,0,1]
        ])

    R = camera_to_worlds[:3, :3]  # 3 x 3
    R_edit = np.diag([1, 1, 1])
    R = R @ R_edit
    R = K@R
    C = camera_to_worlds[:3, 3:4]  # 3 x 1
    t = -C

    X = np.zeros((3,4))
    X[:3, :3] = np.eye(3)
    X[:3, 3:4] = t
    p = R@X

    viewmat = np.eye(4, dtype=R.dtype)
    # viewmat[:3, :3] = R
    # camera_to_worlds[:3, :3] = R
    # viewmat[:3, 3:4] = t
    viewmat[:3, :] = p
    #viewmat = np.linalg.inv(camera_to_worlds)

    return viewmat

input_path= r'C:\Users\MuzaddidMdAhmedAl\docker_mount\outputs\exports\pcd'
dataname="point_cloud.ply"
pcd = o3d.io.read_point_cloud(os.path.join(input_path,dataname))
# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Homogeneous coordinates
points_h = np.hstack((points, np.ones((points.shape[0], 1))))

# Project the points
K = get_intrinsic(640, 480)
P = get_extrinsic(K)
projected_points = P @ points_h.T
projected_points /= projected_points[2, :]  # Normalize

# Extract 2D image coordinates
u = projected_points[0, :].astype(int)
v = projected_points[1, :].astype(int)

import matplotlib.pyplot as plt

plt.scatter(u, v, s=0.1)
plt.show()
