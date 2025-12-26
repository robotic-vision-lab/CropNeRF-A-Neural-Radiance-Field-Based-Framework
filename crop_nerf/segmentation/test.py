# ... having adjusted 't' and 'R' for my desired camera placement and direction
import open3d as o3d
import numpy as np
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

def get_extrinsic(x=0, y=0, z=0, rx=0, ry=0, rz=0):
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = (x, y, z)
    extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([radians(rx), radians(ry), radians(rz)])
    extrinsic= np.asarray ([[1.0, 0.0, 0.0, -0.10035978257656097],
    [-0.0, -1.0, -0.0, -0.07357291877269745],
    [-0.0, -0.0, -1.0, 0.28320024288570145],
    [0.0, 0.0, 0.0, 1.0]])

    extrinsic = np.asarray([[0.046825625, 0.9985399, 0.026933009, 0.047231793],
    [-0.67224306, 0.051444445, -0.73854095, -0.931187],
    [-0.73884815, 0.016477115, 0.6736705, 0.2662813],
                            [0.0, 0.0, 0.0, 1.0]])

    extrinsic = np.asarray([
        [0.046825625,0.9985399,0.026933009, 0.047231793],
[-0.67224306,0.051444445,-0.73854095,-0.931187],
[-0.73884815,0.016477115,0.6736705,0.2662813],
[0,0,0,1]
        ])
    R = extrinsic[:3, :3]
    # R_edit = np.diag([1, -1, -1])
    # R = R @ R_edit
    extrinsic[:3, :3] = R
    return extrinsic

input_path= r'C:\Users\MuzaddidMdAhmedAl\docker_mount\outputs\exports\pcd'
dataname="point_cloud.ply"
pcd = o3d.io.read_point_cloud(os.path.join(input_path,dataname))
vis = o3d.visualization.Visualizer()
vis.create_window(width=1940, height=1440)
vis.add_geometry(pcd)

# Add point cloud to the visualizer
ctr = vis.get_view_control()

width, height= 1940, 1080
intrinsic = get_intrinsic(width, height)
extrinsic = get_extrinsic()

camera_params = o3d.camera.PinholeCameraParameters()
camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic)

camera_params.extrinsic = extrinsic
#camera_params = ctr.convert_to_pinhole_camera_parameters()
ctr.convert_from_pinhole_camera_parameters(camera_params, True)

vis.update_geometry(pcd)
vis.update_renderer()
vis.run()

# doesn't work until after add_geometry...  see Visualizer.cpp line 408 - ResetViewPoint() in Visualizer::AddGeometry
params = ctr.convert_to_pinhole_camera_parameters()
camera_position = params.extrinsic[:3, 3]
print("Default Camera Position:", camera_position)

vis.update_geometry(pcd)
vis.update_renderer()

#
# extrinsic = np.eye(4)
# extrinsic[:3, :3] = R
# extrinsic[:3, 3] = t.flatten()
#
# intrinsic = o3d.camera.PinholeCameraIntrinsic()
# intrinsic.set_intrinsics(cam.width, cam.height, fx, fy, cx, cy)
#
# # Set the camera parameters
# camera_params = o3d.camera.PinholeCameraParameters()
# camera_params.intrinsic = intrinsic
# camera_params.extrinsic = extrinsic
# ctr.convert_from_pinhole_camera_parameters(camera_params, True)
# self.__camera_vis.update_renderer()
