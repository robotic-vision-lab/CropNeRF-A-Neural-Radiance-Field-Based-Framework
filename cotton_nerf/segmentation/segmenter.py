from typing import Counter
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import open3d as o3d
import  os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import SpectralClustering
import sys
from tqdm import tqdm
import  pickle
#from fruit_nerf.clustering import cluster_kmeans, cluster_dbscan, cluster_kmeans_elbow


from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import math

def spectral_clustering(pcd, k=5):
    features = np.asarray(pcd.points)


    clustering = SpectralClustering(n_clusters=k,
                                    affinity= 'nearest_neighbors',
                                    n_neighbors = 8,
                                    assign_labels= 'kmeans', #
                                    random_state=0,
                                    n_jobs=9).fit(features)
    return clustering.labels_

def cluster_kmeans(pcd, k=10, consider_normals=False):

    if consider_normals:
        if not pcd.has_normals():
            pcd.estimate_normals()
        p = np.asarray(pcd.points)
        n = np.asarray(pcd.normals)
        p_norm = (p - p.min(axis=0))/(p.max(axis=0) - p.min(axis=0))
        n_norm =  (n - n.min(axis=0))/(n.max(axis=0) - n.min(axis=0))

        features = np.hstack((p_norm, n_norm, ))
    else:
        features = np.asarray(pcd.points)

    kmeans_cluster = KMeans(init="k-means++", n_clusters=k, n_init='auto', random_state=0)
    kmeans_cluster.fit(features)
    labels = kmeans_cluster.labels_

    return  labels


def crop(pcd):
    min_bound = np.array([-1, -1, -1])
    max_bound = np.array([1, 1, 1])
    translation = (0.0000000000, 0.0000000000, 0.3468994381)
    pcd.translate(translation)

    # Crop the point cloud
    cropped_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
    return cropped_pcd

def show_pcd(pcd, labels):
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


#o3d.visualization.draw_geometries([pcd])

#o3d.visualization.draw_geometries([downpcd])

#downpcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=True)
def get_super_clusters(pcd, vx_size = 10e-5): # 10e-4 for apple,cot, 10e-5 for pear

    downpcd = pcd.voxel_down_sample(voxel_size=vx_size)  # .005
    pcd = downpcd
    '''TODO Experiment Idea: show that the performance does not change based on the parameter, show visually and in tabular format'''
    labels = np.array( pcd.cluster_dbscan(eps=20 * vx_size, min_points=30, print_progress=True))  # eps=0.05, 0.004 [5*-25*]
    non_outlier_mask = labels >= 0
    # outlier based on clustering
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[non_outlier_mask])
    labels = labels[non_outlier_mask]

    #outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    labels = labels[ind]

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    return pcd, labels


def get_nth_largest_super_point(pcd, labels, n):
    largest_cluster_label = sorted([(v,k) for k,v in Counter(labels.tolist()).items()],reverse=True)[n][1]
    largest_subset_pcd = o3d.geometry.PointCloud()
    largest_subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == largest_cluster_label])
    return largest_subset_pcd, labels

def get_dummy_pcd():
    input_path = r'/opt/data/outputs/exports/pcd'
    dataname = "point_cloud.ply"
    pcd = o3d.io.read_point_cloud(os.path.join(input_path, dataname))
    sub_clouds, labels = get_super_clusters(pcd)
    largest_super_point, lables = get_largest_super_point(sub_clouds, labels)
    return largest_super_point, labels

def bounds_as_sorted_list(pcd, labels):
    import time
    largest_cluster_id = sorted([(v,k) for k,v in Counter(labels.tolist()).items()],reverse=True)
    bounds = []
    for i in range(len(largest_cluster_id)):
        id  = largest_cluster_id[i][1]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == id])
        aabb = pc.get_axis_aligned_bounding_box()
        bounds.append(np.stack([aabb.min_bound, aabb.max_bound]))
    return bounds
        #time.sleep(2000)
    # largest_subset_pcd = o3d.geometry.PointCloud()
    # largest_subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == largest_cluster_id])
    # labels = cluster_kmeans(largest_subset_pcd, 5)
    # max_label = labels.max()
    # print(f" subset point cloud has {max_label + 1} clusters")
    #show_pcd(largest_subset_pcd, labe

def get_bounds():
    input_path = r'/opt/data/outputs/exports/pcd'
    dataname = "point_cloud.ply"
    pcd = o3d.io.read_point_cloud(os.path.join(input_path, dataname))
    sub_clouds, labels = get_super_clusters(pcd)
    bounds = bounds_as_sorted_list(sub_clouds, labels)
    return bounds


def process_and_save(pcd_path, nth_largest, k, save_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    sub_clouds, labels = get_super_clusters(pcd)
    pcd_of_interest, lables = get_nth_largest_super_point(sub_clouds, labels, nth_largest)
    #labels = cluster_kmeans(pcd_of_interest, k=n)
    labels = spectral_clustering(pcd_of_interest, k=k)
    pc_aabb = []
    pc_list = []
    res = []
    for i in range(k):
        pc = o3d.geometry.PointCloud()
        pc_as_np = np.asarray(pcd_of_interest.points)[labels == i]
        pc.points = o3d.utility.Vector3dVector(pc_as_np)
        aabb = pc.get_axis_aligned_bounding_box()
        pc_aabb.append(np.stack([aabb.min_bound, aabb.max_bound]))
        pc_list.append(pc_as_np)
    x = {'aabb': np.stack(pc_aabb),
         'pcd': {id:pc for id,pc in enumerate(pc_list)}
         }
    res.append(x)
    np.save( save_path, res)

    print(f'Point cloud saved to {save_path}')


def process_and_save_all(pcd_path, k, save_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd, labels = get_super_clusters(pcd)
    cluster_labels = sorted([(v, k) for k, v in Counter(labels.tolist()).items()], reverse=True)#[n][1]
    res = []
    for n in tqdm(range(len(cluster_labels))):
        label = cluster_labels[n][1]
        pcd_of_interest = o3d.geometry.PointCloud()
        pcd_of_interest.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == label])
        if len(pcd_of_interest.points) <= k:
            continue
        sub_labels = cluster_kmeans(pcd_of_interest, k=k)
        #sub_labels = spectral_clustering(pcd_of_interest, k=k)
        pc_aabb = []
        pc_list = []
        for i in range(k):
            pc = o3d.geometry.PointCloud()
            pc_as_np = np.asarray(pcd_of_interest.points)[sub_labels == i]
            pc.points = o3d.utility.Vector3dVector(pc_as_np)
            aabb = pc.get_axis_aligned_bounding_box()
            pc_aabb.append(np.stack([aabb.min_bound, aabb.max_bound]))
            pc_list.append(pc_as_np)
        x = {'aabb': np.stack(pc_aabb),
             'pcd': {id:pc for id,pc in enumerate(pc_list)}
             }
        res.append(x)
    np.save( save_path, res)

    print(f'Point cloud saved to {save_path}')

def process_for_pipeline(input_path, dataname, k):
    save_path = os.path.join(input_path, f'all_super_cluster_info_nsub_2.npy')
    process_and_save_all(os.path.join(input_path, dataname), k=k, save_path=save_path)

def view_super_clusters(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd, labels = get_super_clusters(pcd)
    show_pcd(pcd, labels)

def view_sub_clusters(pcd_path, super_clusters_idx, k):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd, labels = get_super_clusters(pcd)
    cluster_labels = sorted([(v, k) for k, v in Counter(labels.tolist()).items()], reverse=True)  # [n][1]
    res = []

    label = cluster_labels[super_clusters_idx][1]
    pcd_of_interest = o3d.geometry.PointCloud()
    pcd_of_interest.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == label])
    if len(pcd_of_interest.points) <= k:
        print("Few points")
    labels = cluster_kmeans(pcd_of_interest, k=k)
    show_pcd(pcd_of_interest, labels)


if __name__ == "__main__":
    video = sys.argv[1] #'recording_2024-09-11_12-42-36'
    input_path = rf'C:\Users\MuzaddidMdAhmedAl\docker_mount\artifacts\{video}\pcd'
    input_path = rf'D:\3d_phenotyping\artifacts\{video}\pcd'
    dataname = "semantics_pc.ply" #"semantics_pc_wo_cam_optimize.ply"
    nth = 3

    # save_path = os.path.join(input_path, f'super_cluster_info_{nth}.npy')
    # process_and_save(os.path.join(input_path, dataname), nth_largest=nth, k=6, save_path=save_path)

    #process_for_pipeline(input_path, dataname, k=2)
    view_super_clusters(os.path.join(input_path, dataname))
    #view_sub_clusters(os.path.join(input_path, dataname),  super_clusters_idx=0, k=2)

    # pcd = o3d.io.read_point_cloud(os.path.join(input_path,dataname))
    # sub_clouds, labels = get_sub_clouds(pcd)
    # largest_super_point, lables = get_nth_largest_super_point(sub_clouds, labels, 2)
    # aabb = largest_super_point.get_axis_aligned_bounding_box()
    # print(aabb)
    # labels = cluster_kmeans(largest_super_point, k= 5)
    # show_pcd(largest_super_point, labels)













######################## End of second stage
# pcd = subset_pcd
# c = Counter(labels)
# largest_cluster_id = sorted([(v,k) for k,v in Counter(labels.tolist()).items()],reverse=True)[0][1]
# subset_pcd = o3d.geometry.PointCloud()
# #subset_pcd.points = o3d.utility.Vector3dVector(pcd.points[labels == largest_cluster_id])
# subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == largest_cluster_id])
# labels = np.array(subset_pcd.cluster_dbscan(eps=vx_size, min_points=5, print_progress=True))
# max_label = labels.max()
# print(f" subset point cloud has {max_label + 1} clusters")


