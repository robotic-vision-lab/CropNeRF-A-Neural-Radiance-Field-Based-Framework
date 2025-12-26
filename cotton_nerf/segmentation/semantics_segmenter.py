from typing import Counter
import numpy as np
import open3d as o3d
import  os
import matplotlib.pyplot as plt
from collections import Counter
from clustering import cluster_kmeans, cluster_dbscan

input_path= r'C:\Users\MuzaddidMdAhmedAl\docker_mount\outputs\exports\pcd\fruit_nerf'
dataname='semantic.ply' #"point_cloud.ply"

from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import math

# from pyntcloud import PyntCloud
#
# cloud = PyntCloud.from_file("some_file.ply")

pcd = o3d.io.read_point_cloud(os.path.join(input_path,dataname))
#
# import numpy as np
# import open3d as o3d
# pcd_data = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(pcd_data.path)
# o3d.visualization.draw_geometries([pcd])

def crop(pcd):
    min_bound = np.array([-1, -1, -1])
    max_bound = np.array([1, 1, 1])
    translation = (0.0000000000, 0.0000000000, 0.3468994381)
    pcd.translate(translation)

    # Crop the point cloud
    cropped_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
    return cropped_pcd

#o3d.visualization.draw_geometries([pcd])
vx_size = 1 * 10e-6
downpcd = pcd.voxel_down_sample(voxel_size=vx_size) #.005
#o3d.visualization.draw_geometries([downpcd])
pcd = crop(downpcd)
#o3d.visualization.draw_geometries()

'''TODO Experiment Idea: show that the performance does not change based on the parameter, show visually and in tabular format'''
labels = np.array(pcd.cluster_dbscan(eps=2000*vx_size, min_points=20, print_progress=True)) #eps=0.05, 0.004 [5*-25*]
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
#
############# END of first stage
c = Counter(labels)
largest_cluster_id = sorted([(v,k) for k,v in Counter(labels.tolist()).items()],reverse=True)[0][1]

#pcd= pcd[labels == largest_cluster_id]
subset_pcd = o3d.geometry.PointCloud()
#subset_pcd.points = o3d.utility.Vector3dVector(pcd.points[labels == largest_cluster_id])
subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == largest_cluster_id])
#labels = np.array(subset_pcd.cluster_dbscan(eps=500*vx_size, min_points=5, print_progress=True))
labels = cluster_point_cloud(subset_pcd, eps=10000*vx_size, min_points=10)

max_label = labels.max()
print(f" subset point cloud has {max_label + 1} clusters")
pcd = subset_pcd
#
# ######################## End of second stage
# pcd = subset_pcd
# c = Counter(labels)
# largest_cluster_id = sorted([(v,k) for k,v in Counter(labels.tolist()).items()],reverse=True)[0][1]
# subset_pcd = o3d.geometry.PointCloud()
# #subset_pcd.points = o3d.utility.Vector3dVector(pcd.points[labels == largest_cluster_id])
# subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == largest_cluster_id])
# labels = np.array(subset_pcd.cluster_dbscan(eps=vx_size, min_points=5, print_progress=True))
# max_label = labels.max()
# print(f" subset point cloud has {max_label + 1} clusters")
#
#
#
#
colors = plt.get_cmap("tab20")(labels / (max_label if max_label+1 > 0 else 1))
# col_map = generate_colormap(max_label)
# colors = col_map(labels)
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd])
