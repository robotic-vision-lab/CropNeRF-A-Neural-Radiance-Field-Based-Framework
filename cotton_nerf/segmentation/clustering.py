import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot as plt
import os
from collections import Counter
from sklearn.cluster import AgglomerativeClustering


def show_pcd(pcd, labels=None):
    if labels is not None:
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def cluster_dbscan(pcd, eps=0.1, min_points=10):
    """
    Clusters a point cloud based on location and normals.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        eps (float): The radius for DBSCAN clustering.
        min_points (int): The minimum number of points required to form a cluster.

    Returns:
        list: A list of point cloud clusters.
    """

    # Estimate normals if not already present
    if not pcd.has_normals():
        pcd.estimate_normals()


    #features = np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals)))
    features = np.asarray(pcd.points)
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_points)
    labels = dbscan.fit_predict(features)

    # Extract clusters
    # clusters = []
    # for label in np.unique(labels):
    #     if label != -1:  # Exclude noise points
    #         cluster_indices = np.where(labels == label)[0]
    #         cluster_pcd = pcd.select_by_index(cluster_indices)
    #         clusters.append(cluster_pcd)

    return  labels

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


def cluster_kmeans_elbow(pcd, max_k=10):
    from sklearn.metrics import silhouette_score
    features = np.asarray(pcd.points)
    sil = []


    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    prev_inertia = 1000
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k).fit(features)
        labels = kmeans.labels_
        i = kmeans.inertia_
        print(f"Clustering with k={k}\t inertia={i}")
        if  prev_inertia - i <10:
            return labels
        prev_inertia = i
    return labels

def spectral_clustering(pcd, k=5):
    features = np.asarray(pcd.points)


    clustering = SpectralClustering(n_clusters=k,
                                    affinity= 'nearest_neighbors',
                                    n_neighbors = 4,
                                    assign_labels= 'kmeans', #
                                    random_state=0,
                                    n_jobs=9).fit(features)
    return clustering.labels_


def plot_histogram_from_counter(labels):
    """Plots a histogram from a Counter object."""

    labels, values = zip(*counter.items())
    plt.bar(labels, values)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram from Counter")
    plt.show()

def view_super_point(super_clusters_pcd, labels, nth=0, vx_size = 10e-4):
    downpcd = pcd.voxel_down_sample(voxel_size=vx_size)  # .005
    pcd = downpcd
    # '''TODO Experiment Idea: show that the performance does not change based on the parameter, show visually and in tabular format'''
    # labels = np.array(
    #     pcd.cluster_dbscan(eps=15 * vx_size, min_points=30, print_progress=True))  # eps=0.05, 0.004 [5*-25*]

    largest_cluster_id = sorted([(v, k) for k, v in Counter(labels.tolist()).items()], reverse=True)[nth][1]
    print('cluster id:', largest_cluster_id)
    largest_subset_pcd = o3d.geometry.PointCloud()
    largest_subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(super_clusters_pcd.points)[labels == largest_cluster_id])

    labels  = cluster_kmeans(largest_subset_pcd, k=10)
    #labels = spectral_clustering(largest_subset_pcd, k=10)
    #labels = largest_subset_pcd.cluster_dbscan(eps=2* vx_size, min_points=10, print_progress=True)
    #hierarchical_cluster = AgglomerativeClustering(n_clusters=10, linkage='average')
    #labels = hierarchical_cluster.fit_predict(np.asarray(largest_subset_pcd.points))

    labels = np.asarray(labels)
    show_pcd(largest_subset_pcd, labels)

    return labels


def get_super_cluster(pcd, vx_size =  10e-4, ): #5 * 10e-5
    downpcd = pcd.voxel_down_sample(voxel_size=vx_size)  # .005
    #pcd = downpcd
    '''TODO Experiment Idea: show that the performance does not change based on the parameter, show visually and in tabular format'''
    labels = np.array(
        pcd.cluster_dbscan(eps=20 * vx_size, min_points=30, print_progress=True))  # eps=0.05, 0.004 [5*-25*]
    non_outlier_mask = labels>=0
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[non_outlier_mask])
    labels = labels[non_outlier_mask]
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    labels = labels[ind]

    return pcd, labels



# Example usage:
if __name__ == "__main__":
    input_path = r'D:\3d_phenotyping\artifacts\recording_2024-09-11_12-31-01\pcd'
    dataname = "semantics_pc.ply"
    #dataname = 'full_tree_pc.ply' #"full_tree_seg_result.ply"
    pcd = o3d.io.read_point_cloud(os.path.join(input_path, dataname))
    super_cluster, labels = get_super_cluster(pcd)
    show_pcd(super_cluster, labels)
    #view_super_point(super_cluster, labels, 3)
    #show_pcd(super_cluster, labels) #[121,223,5]
    # pcd.colors = o3d.utility.Vector3dVector([ [.5,.5,.5] if  p[-1]>-.7  else [.5, .5, .5]  for p in np.asarray(pcd.points)])
    # show_pcd(pcd)
    #view_super_point(super_cluster, labels)
