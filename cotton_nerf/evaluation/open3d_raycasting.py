import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os
import time
cube = o3d.t.geometry.TriangleMesh.from_legacy(
                                    o3d.geometry.TriangleMesh.create_box())

# Create scene and add the cube mesh
scene = o3d.t.geometry.RaycastingScene()
#scene.add_triangles(cube)
cx=962.27
cy = 722.84
fx = 1444.23
fy = 1444.23

c2w = np.asarray( [[-0.02684204,0.99940807,0.021519292,-0.22928147],
[-0.6656043,-0.0018073402,-0.74630266,-0.8462026],
[-0.745822,-0.034355618,0.66525877,0.10826472]])




def get_projection_mat(fx, fy, cx, cy, c2w):  # 3D points
    orig = c2w[:3, 3]
    rot_inv = c2w[:3, :3].T
    t = -rot_inv @ orig

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot_inv
    extrinsic[:3, 3] = t

    K = np.asarray([[fx, 0, -cx, 0],
                    [0, -fy, -cy, 0],
                    [0, 0, -1, 0]])
    P = K @ extrinsic  # a[:3,:]

    return P, K[:,:3], extrinsic

def get_super_cluster(super_cluster_info, idx):
    sup_cluster = super_cluster_info[idx]
    sup_pc = sup_cluster['pcd']
    cluster_pc = np.vstack([pc for _,pc in sup_pc.items()])
    return cluster_pc




def get_sub_cloud_projection(P, sub_pc):
    pass

# Rays are 6D vectors with origin and ray direction.
# Here we use a helper function to create rays for a pinhole camera.
# rays = scene.create_rays_pinhole(fov_deg=60,
#                                  center=[0.5,0.5,0.5],
#                                  eye=[-1,-1,-1],
#                                  up=[0,0,1],
#                                  width_px=320,
#                                  height_px=240)

P, K, E = get_projection_mat(fx, fy, cx, cy, c2w)

input_path= r'D:\3d_phenotyping\artifacts\recording_2024-09-11_12-14-59\pcd'
dataname="semantics_pc.ply"
cluster_info_path = os.path.join(input_path, 'all_super_cluster_info.npy')
cluster_info = np.load(cluster_info_path, allow_pickle=True)
cluster = get_super_cluster(cluster_info, 0)









device = o3d.core.Device("CPU:0")
pcd = o3d.io.read_point_cloud(os.path.join(input_path, dataname))
pcd = pcd.voxel_down_sample(voxel_size=10e-4)
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist

start = time.time()

#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, width=.5)[0]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=.01)
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#            pcd,  o3d.utility.DoubleVector([radius, radius * 2]))
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene.add_triangles(mesh)

rays = scene.create_rays_pinhole(intrinsic_matrix=K, extrinsic_matrix=E, width_px=1920, height_px=1440)
res = scene.cast_rays(rays)
dis = res['t_hit'].numpy()
valid_rays = dis < float('inf')
end = time.time()
print(end - start)

# Compute the ray intersections.
#ans = scene.cast_rays(rays)

img = np.zeros(dis.shape +(3,))
img[valid_rays] = 255
#Visualize the hit distance (depth)
plt.imshow(img)
plt.show()















# import open3d as o3d
# import numpy as np
#
# # Create scene and add the monkey model.
# scene = o3d.t.geometry.RaycastingScene()
# d = o3d.data.MonkeyModel()
# mesh = o3d.t.io.read_triangle_mesh(d.path)
# mesh_id = scene.add_triangles(mesh)
#
# # Create a grid of rays covering the bounding box
# bb_min = mesh.vertex['positions'].min(dim=0).numpy()
# bb_max = mesh.vertex['positions'].max(dim=0).numpy()
# x,y = np.linspace(bb_min, bb_max, num=10)[:,:2].T
# xv, yv = np.meshgrid(x,y)
# orig = np.stack([xv, yv, np.full_like(xv, bb_min[2]-1)], axis=-1).reshape(-1,3)
# dest = orig + np.full(orig.shape, (0,0,2+bb_max[2]-bb_min[2]),dtype=np.float32)
# rays = np.concatenate([orig, dest-orig], axis=-1).astype(np.float32)
#
# # Compute the ray intersections.
# lx = scene.list_intersections(rays)
# lx = {k:v.numpy() for k,v in lx.items()}
#
# # Calculate intersection coordinates using the primitive uvs and the mesh
# v = mesh.vertex['positions'].numpy()
# t = mesh.triangle['indices'].numpy()
# tidx = lx['primitive_ids']
# uv = lx['primitive_uvs']
# w = 1 - np.sum(uv, axis=1)
# c = \
# v[t[tidx, 1].flatten(), :] * uv[:, 0][:, None] + \
# v[t[tidx, 2].flatten(), :] * uv[:, 1][:, None] + \
# v[t[tidx, 0].flatten(), :] * w[:, None]
#
# # Calculate intersection coordinates using ray_ids
# c = rays[lx['ray_ids']][:,:3] + rays[lx['ray_ids']][:,3:]*lx['t_hit'][...,None]
#
# # Visualize the rays and intersections.
# lines = o3d.t.geometry.LineSet()
# lines.point.positions = np.hstack([orig,dest]).reshape(-1,3)
# lines.line.indices = np.arange(lines.point.positions.shape[0]).reshape(-1,2)
# lines.line.colors = np.full((lines.line.indices.shape[0],3), (1,0,0))
# x = o3d.t.geometry.PointCloud(positions=c)
# o3d.visualization.draw([mesh, lines, x], point_size=8)