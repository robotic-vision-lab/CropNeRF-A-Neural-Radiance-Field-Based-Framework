import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot as plt
import os
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
import matplotlib
from matplotlib import cm, colors
cmap = matplotlib.colors.ListedColormap(cm.tab20.colors + cm.tab20c.colors, name='tab40')

def show_pcd(pcd, labels=None, full_tree=None, box=None):
    if box is None:
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pcd.get_axis_aligned_bounding_box())
    box.paint_uniform_color([.765, .129, .282])


    if labels is not None:
        max_label = labels.max()
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors = cmap(labels)# / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #vis.add_geometry(box)

    if full_tree:
        full_tree = full_tree.voxel_down_sample(voxel_size=5*10e-4)
        n_points = len(full_tree.points)
        full_tree.colors = o3d.utility.Vector3dVector(np.ones((n_points,3))*[255/255,255/255,255/255])
        vis.add_geometry(full_tree)
    vis.add_geometry(pcd)

    vis.run()
    return pcd


def view_semantic_cloud(basedir, recording, vis_full_tree):
    tree_pcd_file = 'full_tree_pc.ply'
    semantic_pcd_file = 'semantics_pc.ply'
    pcd_tree = o3d.io.read_point_cloud(os.path.join(basedir, recording, 'pcd', tree_pcd_file))
    pcd_semantic = o3d.io.read_point_cloud(os.path.join(basedir, recording, 'pcd', semantic_pcd_file))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_semantic)

    if vis_full_tree:
        pcd_tree = o3d.io.read_point_cloud(os.path.join(basedir, recording, 'pcd', tree_pcd_file))
        pcd_tree = pcd_tree.voxel_down_sample(voxel_size=5 * 10e-4)
        n_points = len(pcd_tree.points)
        #pcd_tree.colors = o3d.utility.Vector3dVector(np.ones((n_points, 3)) * [1 / 255, 211 / 255, 100 / 255])
        pcd_tree.colors = o3d.utility.Vector3dVector(np.ones((n_points, 3)) * [255 / 255, 255 / 255, 255 / 255])
        vis.add_geometry(pcd_tree)

    vis.run()










def view_super_clusters(basedir, recording):
    input_path = os.path.join(basedir, recording, 'pcd',  'all_super_cluster_info.npy' )
    tree_pcd_file = 'full_tree_pc.ply'
    pcd_tree = o3d.io.read_point_cloud(os.path.join(basedir, recording, 'pcd', tree_pcd_file))
    pcd_data = np.load(input_path,  allow_pickle=True)
    n_super_clusters = len(pcd_data)

    labels = []
    points = []
    points_to_highlight= []

    for idx in range(n_super_clusters):

        sub_clusters = pcd_data[idx]['pcd']
        n_sub_clusters = len(sub_clusters)
        for i in range(n_sub_clusters):
            if idx == 1:
                points_to_highlight.append(sub_clusters[i])
            n_point = len(sub_clusters[i])
            points.append(sub_clusters[i])
            labels.extend([idx+1 for _ in range(n_point)])

    # pcd_of_interest = o3d.geometry.PointCloud()
    # pcd_of_interest.points = o3d.utility.Vector3dVector(np.vstack(points_to_highlight))
    # highlight_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pcd_of_interest.get_axis_aligned_bounding_box())
    # colors = cmap(2*np.ones(len(pcd_of_interest.points)) / 16)
    # pcd_of_interest.colors =  o3d.utility.Vector3dVector(colors[:, :3])
    # show_pcd(pcd_of_interest,None, None, highlight_box)

    points = np.concatenate(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
    #pcd = pcd.crop(pcd_of_interest.get_axis_aligned_bounding_box())
    show_pcd(pcd, np.asarray(labels), pcd_tree, None)


def view_sub_clusters_with_camera(basedir, recording, idx):
    input_path = os.path.join(basedir, recording, 'pcd',  'all_super_cluster_info.npy' )
    tree_pcd_file = 'full_tree_pc.ply'
    pcd_tree = o3d.io.read_point_cloud(os.path.join(basedir, recording, 'pcd', tree_pcd_file))
    pcd_data = np.load(input_path,  allow_pickle=True)
    sub_clusters = pcd_data[idx]['pcd']
    n_sub_clusters = len(sub_clusters)
    pcd = o3d.geometry.PointCloud()
    labels = []
    points = []
    l =1
    c = [26,8]
    for i in [7]:#range(n_sub_clusters):
        n_point = len(sub_clusters[i])
        points.append(sub_clusters[i])
        labels.extend([c[l] for _ in range(n_point)])
        l+=1
    points = np.concatenate(points)
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points)  * np.array([10., 10., 10.]))
    pcd = pcd.scale(1,[0,0,0])
    highlight_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box( pcd.get_axis_aligned_bounding_box())

    highlight_box.paint_uniform_color([.765, .129, .282])
    colors = cmap(labels)  # / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #vis.add_geometry(highlight_box)
    vis.add_geometry(pcd)

    cam1 = np.asarray([[1.0,0.0,0.0,0.20463307201862335],
                         [-0.0,-1.0,-0.0,0.010784770051638283],
                    [-0.0,-0.0,-1.0,-.375650557329989],
                          [0.0,0.0,0.0,1.0]])
    cam1_intrinsic = np.asarray([[1,0.0,1.5],
[0.0,1,1.0],
[0.0,0.0,1.0]]
)

    # add camera
    # standardCameraParametersObj = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # print(standardCameraParametersObj.extrinsic)
    # #cameraLines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=standardCameraParametersObj.intrinsic, extrinsic=cam1)
    # cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=3, view_height_px=2,
    #                                                                   intrinsic=cam1_intrinsic,
    #                                                                   extrinsic=cam1)
    # vis.add_geometry(cameraLines)

    vis.run()




def view_sub_clusters(basedir, recording, idx):
    input_path = os.path.join(basedir, recording, 'pcd',  'all_super_cluster_info.npy' )
    tree_pcd_file = 'full_tree_pc.ply'
    pcd_tree = o3d.io.read_point_cloud(os.path.join(basedir, recording, 'pcd', tree_pcd_file))
    pcd_data = np.load(input_path,  allow_pickle=True)
    sub_clusters = pcd_data[idx]['pcd']
    n_sub_clusters = len(sub_clusters)
    pcd = o3d.geometry.PointCloud()
    labels = []
    points = []
    for i in range(n_sub_clusters):
        n_point = len(sub_clusters[i])
        points.append(sub_clusters[i])
        labels.extend([i+1 for _ in range(n_point)])
    points = np.concatenate(points)
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
    highlight_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box( pcd.get_axis_aligned_bounding_box())

    show_pcd(pcd, np.asarray(labels), None, highlight_box)


def view_final_instances(basedir, recording):
    input_path = os.path.join(basedir, recording, 'pcd')#r'D:\3d_phenotyping\artifacts\recording_2024-09-11_12-31-01\pcd'
    sem_pcd_file = "full_tree_seg_result.ply"  # "semantics_pc.ply"
    tree_pcd_file = 'full_tree_pc.ply'  # "full_tree_seg_result.ply"
    pcd_tree = o3d.io.read_point_cloud(os.path.join(input_path, tree_pcd_file))
    pcd_seg = o3d.io.read_point_cloud(os.path.join(input_path, sem_pcd_file))
    show_pcd(pcd_seg, full_tree=pcd_tree)

if __name__ == "__main__":
    basedir = r'D:\3d_phenotyping\artifacts'
    recording = 'recording_2024-09-11_12-31-01'
    #view_semantic_cloud(basedir, recording, vis_full_tree=False)
    #view_final_instances(basedir, recording)
    #view_sub_clusters(basedir, recording, 1)
    view_super_clusters(basedir, recording)
    #view_sub_clusters_with_camera(basedir, recording, 0)



    # input_path = r'D:\3d_phenotyping\artifacts\recording_2024-09-11_12-31-01\pcd'
    # sem_pcd_file = "full_tree_seg_result.ply"#"semantics_pc.ply"
    # tree_pcd_file = 'full_tree_pc.ply' #"full_tree_seg_result.ply"
    # pcd_tree = o3d.io.read_point_cloud(os.path.join(input_path, tree_pcd_file))
    # pcd_seg = o3d.io.read_point_cloud(os.path.join(input_path, sem_pcd_file))
    # #super_cluster, labels = get_super_cluster(pcd)
    # #show_pcd(super_cluster, labels) #[121,223,5]
    # #pcd_seg.colors = o3d.utility.Vector3dVector([[1, 165/255, 0] for p in np.asarray(pcd_seg.points)])
    # show_pcd(pcd_seg, full_tree=pcd_tree)
    # #view_super_point(super_cluster, labels)
