import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from math import radians

cx=962.27
cy = 722.84
fx = 1444.23
fy = 1444.23
#
# x_img -> x_camera
# x_cam = (x_img - cx)/fx
# y_cam = (y_img - cy)/fy * -1
#x_camera ->  x_image_coordinate
#x_img = (x_cam * fx) +cx
# points_cam = np.sum(points[:, None, :] *rot_T, axis=-1) +t
# points_cam = points_cam /-points_cam[:,-1:]
# points_img = points_cam[:,:2] *[fx,-fy] + [cx, cy]
# points_img = (K[:3,:3]@points_cam.T).T



c2w = np.asarray( [[-0.02684204,0.99940807,0.021519292,-0.22928147],
[-0.6656043,-0.0018073402,-0.74630266,-0.8462026],
[-0.745822,-0.034355618,0.66525877,0.10826472]])



input_path= r'D:\3d_phenotyping\artifacts\recording_2024-09-11_12-14-59\pcd'
dataname="semantics_pc.ply"
pcd = o3d.io.read_point_cloud(os.path.join(input_path,dataname))
# Convert point cloud to numpy array
points = np.asarray(pcd.points)

#points_cam = np.sum(points[:, None, :] *rot_T, axis=-1) - c2w[:3,3]


#points[:,-1]*=-1
points_h = np.hstack((points, np.ones((points.shape[0], 1))))

def get_projection_mat(fx, fy, cx, cy, c2w): # 3D points
     orig = c2w[:3, 3]
     rot_inv = c2w[:3, :3].T
     t = -rot_inv @ orig

     extrinsic = np.eye(4)
     extrinsic[:3, :3] = rot_inv
     extrinsic[:3, 3] = t

     K = np.asarray([[fx, 0, -cx, 0],
                     [0, -fy, -cy, 0],
                     [0, 0, 1, 0]])
     P = K @ extrinsic  # a[:3,:]

     return P

def get_projection(P, points):
     points_h = np.hstack((points, np.ones((points.shape[0], 1))))
     im = P @ points_h.T
     im = im.T
     im /= -im[:, 2:3]
     return im



def super_point_projection(camera_mat):
     cluster_data = np.load(os.path.join(input_path, 'all_super_cluster_info.npy'), allow_pickle=True)
     n_super_clusters = len(cluster_data)
     for i_sc in range(n_super_clusters):
          cluster_aabb = cluster_data[i_sc]['aabb']
          super_points = cluster_data[0]['pcd']
          img = np.zeros((1440, 1920, 3), dtype=np.uint8)
          for sp_id, sp in super_points.items():
               im = get_projection(camera_mat, sp)
               u = im[:, 0].astype(int)
               v = im[:, 1].astype(int)


               img[v, u] = 255
          plt.imshow(img)
          # plt.scatter(u, v, s=0.1)
          plt.show()



P = get_projection_mat(fx, fy, cx, cy, c2w)
super_point_projection(P)
#


# Extract 2D image coordinates
