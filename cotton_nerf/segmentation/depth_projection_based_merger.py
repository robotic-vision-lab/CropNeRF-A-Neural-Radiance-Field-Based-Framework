import os
from collections import Counter

import numpy as np
import cv2
import glob
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import shutil
import os
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
import networkx as nx
from matplotlib import cm, colors
import  matplotlib
from concurrent.futures import ThreadPoolExecutor

eps = 1e-6

############################################# Function for viewing point cloud #########################################
def get_component(affinity, algo):

    G = nx.from_numpy_array(affinity)
    components = []

    labels = np.zeros(G.order())
    l = 1
    if algo=='clique':
        while G.order()>0:
            # Find a maximal clique
            clique = max(nx.find_cliques(G), key=len)
            components.append(clique)
            # Remove the nodes in the clique from the remaining nodes
            G.remove_nodes_from(clique)
            labels[clique] = l
            l += 1
    elif algo=='bridge':
        G_sub = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in G_sub:
            if len(g)>2: #check for bridge edge
                for e in nx.bridges(g):
                    g.remove_edge(*e)

            for c in nx.connected_components(g):
                if len(c)==1:
                    labels[list(c)] = 0
                    continue
                components.append(c)
                labels[list(c)] = l
                l+=1
    elif algo=='community':
        coms = list(nx.algorithms.community.asyn_lpa_communities(G, weight='weight'))
        for c in coms:
            c = list(c)
            components.append(c)
            labels[c] = l
            l += 1

    return len(components), labels

def draw_graph_from_adjacency_matrix(adj_matrix):
    """
    Draws a graph from a given adjacency matrix.
    """

    G = nx.from_numpy_array(adj_matrix)
    nx.draw(G, with_labels=True)
    plt.show()

def show_pcd(pcd, labels, full_tree=None):
    cmap = matplotlib.colors.ListedColormap(cm.tab20.colors + cm.tab20c.colors, name='tab40')
    max_label = labels.max()
    #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors =cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if full_tree:
        full_tree = full_tree.voxel_down_sample(voxel_size=5*10e-4)
        n_points = len(full_tree.points)
        full_tree.colors = o3d.utility.Vector3dVector(np.ones((n_points,3))*[1/255,211/255,100/255])
        vis.add_geometry(full_tree)
    vis.add_geometry(pcd)

    #o3d.io.write_point_cloud('test_pcd.ply', pcd)
    #o3d.visualization.draw_geometries([pcd])
    #vis.add_3d_label(np.asarray([.5,.5,.5]), 'hello')
    vis.run()

    return pcd

def create_pcd_with_labels(pcd_data,super_cluster_id, super_point_labels):
    superpoint_id_to_pcd = pcd_data[super_cluster_id]['pcd']
    points = []
    labels = []
    for cluster_idx in superpoint_id_to_pcd:
        #if cluster_idx == 5: continue
        p = superpoint_id_to_pcd[cluster_idx]
        l = super_point_labels[cluster_idx]
        points.append(p)
        labels.append(np.ones(len(p))*l)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
    return pcd, np.concatenate(labels)

######################################################## Function to draw overly cand coping label images###############
# def overly_mask_with_projection(projection_dir, label_dir, args):
#     print("Generating overly and coping label frames")
#     cam_dirs = [os.path.basename(x) for x in glob.glob(f"{projection_dir}/cam_*")]
#     overlay_dir = os.path.join(projection_dir, 'overlay')
#     os.makedirs(overlay_dir, exist_ok=True)
#     for cam_dir in tqdm(cam_dirs):
#         subdir = os.path.join(projection_dir, cam_dir)
#         img_name = os.path.split(glob.glob(os.path.join(subdir,'frame*.png'))[0])[-1]
#         segment_img_path = os.path.join(subdir, img_name)
#         frame_label_path =  os.path.join(label_dir, f'label_{img_name}')
#         shutil.copy(frame_label_path, subdir)
#
#         # overlay projection on top of the segmentation image for debuging purpose
#         projection_img_files = glob.glob(os.path.join(subdir, f'{args.visible_img_prefix}*.png'))
#         seg_frame = cv2.imread(segment_img_path)
#         merged = np.zeros_like(seg_frame, dtype=np.uint8)
#
#         for i in range(len(projection_img_files)):
#             proj_img = cv2.imread(projection_img_files[i])
#             mask = proj_img.astype(bool)
#             merged[mask] = proj_img[mask]
#
#         overlayed_img = cv2.addWeighted(seg_frame, .5, merged, .5, 0)
#         cv2.imwrite(segment_img_path, overlayed_img)
#         cv2.imwrite(os.path.join(overlay_dir, f'label_{img_name}'), overlayed_img)

def overly_mask_with_projection(projection_dir, label_dir, args):
    print("Generating overly and coping label frames")
    cam_dirs = [os.path.basename(x) for x in glob.glob(f"{projection_dir}/cam_*")]
    overlay_dir = os.path.join(projection_dir, 'overlay')
    os.makedirs(overlay_dir, exist_ok=True)
    for cam_dir in tqdm(cam_dirs):
        subdir = os.path.join(projection_dir, cam_dir)
        img_name = os.path.split(glob.glob(os.path.join(subdir,'frame*.png'))[0])[-1]
        segment_img_path = os.path.join(subdir, img_name)
        frame_label_path =  os.path.join(label_dir, f'label_{img_name}')
        shutil.copy(frame_label_path, subdir)

        # overlay projection on top of the segmentation image for debuging purpose
        projection_img_file = glob.glob(os.path.join(subdir, 'visible*.png'))[0]
        seg_frame = cv2.imread(segment_img_path)
        merged = np.zeros_like(seg_frame, dtype=np.uint8)

        #for i in range(len(projection_img_files)):
        proj_img = cv2.imread(projection_img_file)
        mask = proj_img.astype(bool)
        merged[mask] = proj_img[mask]

        overlayed_img = cv2.addWeighted(seg_frame, .5, merged, .5, 0)
        cv2.imwrite(segment_img_path, overlayed_img)
        cv2.imwrite(os.path.join(overlay_dir, f'label_{img_name}'), overlayed_img)

#####################################  Core functions to calculte overlaps and affinity ##############################
def get_visible_projection_area(args, camdir, cid,  bbox_xyxy=None, thres=127):
    visible_projection_path = os.path.join(camdir, f'{args.visible_img_prefix}_{cid}.png')
    segment_label_path = glob.glob(os.path.join(camdir, 'label_frame*.png'))[0]
    img = cv2.imread(visible_projection_path)[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded_img = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    # countour based method won't work, since due to illution projection might be separated into multiple contour
    contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return eps, 0, eps
    cnt = max(contours, key=cv2.contourArea)

    segment_mask = np.zeros_like(thresholded_img)
    #thresholded_img.astype(bool)
    cv2.drawContours(segment_mask, cnt, contourIdx=-1, color=255, thickness=-1)
    segment_mask = segment_mask.astype(bool)
    area = segment_mask.sum()
    if area < 10:
        return eps, 0, eps

    # finding label
    segment_label = cv2.imread(segment_label_path, cv2.IMREAD_GRAYSCALE)[bbox_xyxy[1]:bbox_xyxy[3],
                    bbox_xyxy[0]:bbox_xyxy[2]]

    labels = segment_label[segment_mask]

    labels = sorted([(v,k) for k,v in Counter(labels).items()], reverse=True)

    #assert len(labels) <= 2, 'A super point should only have 1 label and 1 background label'
    # segmentation label that overlaps with the maximum numbers of pixel of the
    # projection is assigned as the super point label
    label_area, label  = labels[0]

    label_area = 0 if label == 0 else label_area # area is 0 if label is background

    return area, label, label_area

def get_wo_occlusion_projection_area(fname, thres):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded_img = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # take the first contour
    if len(contours) == 0:
        return eps, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 10:
        return eps, None

    # compute the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_xyxy = (x, y, x + w, y + h)

    return area, bbox_xyxy

def process_super_cluster(projection_dir, n_super_points, binary_thresh, args):
    cams_dir = [os.path.basename(x) for x in glob.glob(f"{projection_dir}/cam_*")]
    superpoint_id = [i for i in range(n_super_points)]
    cluster_prop = {}
    n_cams = len(cams_dir)

    with tqdm(total=n_cams * n_super_points) as pbar:
        for cid in superpoint_id:
            vis_area = eps*np.ones(n_cams)
            wo_occ_area = eps*np.ones(n_cams)
            vis_segmentaion_overlap_area = eps*np.ones(n_cams)
            vis_segmentaion_overlap_label = np.zeros(n_cams)

            for cam_dir_name in cams_dir[::args.frame_sampling_interval]:
                pbar.update(1)
                cam_id = int(cam_dir_name.split('_')[-1])
                camdir = os.path.join(projection_dir, cam_dir_name)

                wo_occlusion_projection_path = os.path.join(camdir, f'{args.wo_occ_img_prefix}_{cid}.png')
                area, bbox_xyxy = get_wo_occlusion_projection_area(wo_occlusion_projection_path, thres=binary_thresh)
                wo_occ_area[cam_id] = area
                if area == eps:
                    vis_area[cam_id] = area
                    continue
                # Apply simple thresholding

                #area, label, label_overlap_area = get_visible_projection_area(visible_projection_path, segment_label_path, bbox_xyxy, thres=binary_thresh)
                area, label, label_overlap_area = get_visible_projection_area(args, camdir, cid, bbox_xyxy,
                                                                              thres=binary_thresh)

                vis_area[cam_id] = area
                vis_segmentaion_overlap_area[cam_id] = label_overlap_area
                vis_segmentaion_overlap_label[cam_id] = label

            wo_occ_area_norm = wo_occ_area / wo_occ_area.max()
            if args.area_normalize:
                reliability = wo_occ_area_norm * (vis_segmentaion_overlap_area / wo_occ_area)
            else:
                reliability = vis_segmentaion_overlap_area / wo_occ_area

            cluster_prop[cid] = {'visible_area': vis_area,
                                'wo_occ_area': wo_occ_area,
                                 'wo_occ_area_norm':  wo_occ_area_norm,
                                'label': vis_segmentaion_overlap_label,
                            'label_overlap_area': vis_segmentaion_overlap_area,
                             'reliability': reliability
                            }

    return cluster_prop

def calc_affinity(cluser_prop):
    n_clusters = len(cluser_prop)
    affinity = np.zeros((n_clusters, n_clusters))

    for i in range(len(cluser_prop)):
        i_label = cluser_prop[i]['label']
        i_reliability = cluser_prop[i]['reliability']

        for j in range(i+1, n_clusters):
            j_label = cluser_prop[j]['label']
            j_reliability = cluser_prop[j]['reliability']

            same_label = (i_label == j_label) & (i_label!=0) & (j_label!=0) #i_label == j_label
            dif_label = (i_label != j_label) & (i_label!=0) & (j_label!=0)
            a_pos = i_reliability[same_label] @ j_reliability[same_label]
            a_neg = i_reliability[dif_label] @ j_reliability[dif_label]

            affinity[i, j] = a_pos - a_neg #1 if a_pos - a_neg >0 else 0
            affinity[j, i] = affinity[i, j]

    return affinity

# def infer_count(arg):

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--base_dir', type=str, required=False, default=r'C:\Users\MuzaddidMdAhmedAl\docker_mount')
    parser.add_argument('--base_dir', type=str, required=False, default=r'D:\3d_phenotyping') #D:\3d_phenotyping
    parser.add_argument('--recording_name', type=str, required=False, default='recording_2024-09-11_12-14-59')
    parser.add_argument('--visible_img_prefix', type=str, required=False, default='visible_cluster')
    parser.add_argument('--wo_occ_img_prefix', type=str, required=False, default='wo_occ_cluster')

    parser.add_argument('--area_normalize', type=bool, required=False, default=False)
    parser.add_argument('--visualize', type=bool, required=False, default=True)
    parser.add_argument('--vis_full_tree', type=bool, required=False, default=True)
    parser.add_argument('--graph_partition', type=str, required=False, default='community')

    parser.add_argument('--super_cluster_idx', type=int, required=False, default=0)  # -1 to consider all
    parser.add_argument('--binary_threshold', type=int, required=False, default=100)
    parser.add_argument('--frame_sampling_interval', type=int, required=False, default=1)
    parser.add_argument('--n_thread', type=int, required=False, default=10)

    args = parser.parse_args()

    projection_dir = os.path.join(args.base_dir,'artifacts', args.recording_name, 'depth_projection')
    pcd_dir = os.path.join(args.base_dir,'artifacts', args.recording_name, 'pcd')
    label_dir = os.path.join(args.base_dir,'training_data', args.recording_name, 'SegmentationLabel')
    pcd_path = os.path.join(pcd_dir, 'all_super_cluster_info.npy' )
    pcd_data = np.load(pcd_path, allow_pickle=True)
    n_super_clusters = len(pcd_data)
    super_point_per_cluster = pcd_data[0]['aabb'].shape[0]

    def handle_single_super_cluster(super_cluster_idx):
        super_cluster_dir = os.path.join(projection_dir, f'super_cluster_{super_cluster_idx}')

        if not os.path.exists(os.path.join(super_cluster_dir, 'overlay')):
            overly_mask_with_projection(super_cluster_dir, label_dir, args)

        cluster_prop = process_super_cluster(super_cluster_dir,
                                             super_point_per_cluster,
                                             args.binary_threshold, args)
        affinity = calc_affinity(cluster_prop)
        # np.set_printoptions(precision=3, suppress=True)
        # print(affinity)

        '''Find the number of connected components'''
        norm_affinity = affinity / np.abs(affinity.max(axis=1, keepdims=True))
        #norm_affinity = np.where(np.abs(norm_affinity) < .3, 0, norm_affinity)
        #affinity = np.where(affinity > 0, 1, 0)
        n_components, labels = get_component(norm_affinity, args.graph_partition)

        # sparse_matrix = csr_matrix(affinity)
        # n_components, labels = connected_components(sparse_matrix)
        #print("number of components:", n_components, "\nlabels:", labels)
        return n_components, labels, affinity

    '''consider single super cluster'''
    n_bolls_list = []
    labels_list = []
    total_boll = 0
    affinity_list = []

    if args.super_cluster_idx != -1:
        n_boll, labels, af = handle_single_super_cluster(args.super_cluster_idx)
        n_bolls_list.append(n_boll)
        affinity_list.append(af)

        pcd, l = create_pcd_with_labels(pcd_data, args.super_cluster_idx, labels)
        labels_list.append(l)
        total_boll += n_boll
    else:
        pcd = o3d.geometry.PointCloud()
        sc_ids = [i for i in range(min(4, n_super_clusters))]
        with ThreadPoolExecutor(max_workers=args.n_thread) as executor:
            outputs = list(executor.map(handle_single_super_cluster, sc_ids))

        for i, (n_boll, labels, af) in enumerate(outputs):
            print(f'{i}_th super cluster has: {n_boll}.')
            labels = [l + total_boll for l in
                      labels]  # to shift the label to have unique lables over all the super cluster
            p, l = create_pcd_with_labels(pcd_data, i, labels)

            n_bolls_list.append(n_boll)
            labels_list.append(l)
            affinity_list.append(af)
            pcd += p
            total_boll += n_boll

    print(f"Total bool: {total_boll}")
    if args.visualize:
        full_tree = None
        if args.vis_full_tree:
            pc_path = os.path.join(args.base_dir, 'artifacts', args.recording_name, 'pcd', 'full_tree_pc.ply')
            full_tree = o3d.io.read_point_cloud(pc_path)

        if len(affinity_list) ==1:
            draw_graph_from_adjacency_matrix(affinity_list[0])
        pcd = show_pcd(pcd, np.concatenate(labels_list), full_tree)
        if args.super_cluster_idx == -1:
            o3d.io.write_point_cloud(os.path.join(pcd_dir, "full_tree_seg_result.ply"), pcd)


if __name__ == '__main__':
    main()
