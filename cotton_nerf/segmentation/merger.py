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
#from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import cm, colors
import matplotlib
from concurrent.futures import ThreadPoolExecutor
cmap = matplotlib.colors.ListedColormap(cm.tab20.colors + cm.tab20c.colors, name='tab40')
import lpa


eps = 1e-6
graph_seed =35
############################################# Function for viewing point cloud #########################################
def get_component(affinity, algo, draw=False):

    if algo == 'clique' or algo == 'bridge':
        affinity = np.where(affinity > 0, 1, 0)
    G = nx.from_numpy_array(affinity)
    #pos = nx.spring_layout(G, seed=graph_seed)  # setting the positions with respect to G, not k.
    pos = nx.circular_layout(G)
    components = []
    colors = cmap(np.arange(1,affinity.shape[0]+1) / affinity.shape[0])

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
        #coms = list(nx.algorithms.community.asyn_lpa_communities(G, weight='weight')) #asyn_lpa_communities
        coms = list(lpa.asyn_lpa_communities(G, weight='weight')) #asyn_lpa_communities
        for c in coms:
            c = list(c)
            components.append(c)
            labels[c] = l
            l += 1
            if draw:
                k = G.subgraph(c)
                nx.draw_networkx(k, node_size=800, pos=pos, node_color=colors[c], edge_color='g', font_weight='bold')
        if draw:
            plt.savefig('segmented_instances_graph.png')
            plt.show()
    return len(components), labels



def draw_graph_from_adjacency_matrix(adj_matrix):
    """
    Draws a graph from a given adjacency matrix.
    """
    n_node = adj_matrix.shape[0]
    node_colors = cmap(np.arange(1, n_node+1)) #/ len(adj_matrix))
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.circular_layout(G)#nx.spring_layout(G, seed=graph_seed) #graphviz_layout(G) #
    #nx.draw(G, pos, with_labels=True)
    #nx.draw(G, pos, node_size=900, node_color=node_colors, with_labels=True) #plt.cm.Dark2

    #labels = nx.get_edge_attributes(G, 'weight')
    # labels = {k:f'{v:.2f}' for k, v in labels.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    edges = G.edges()
    edge_colors = ['g' if G[u][v]['weight']>0 else 'r' for u, v in edges]
    weights = [10*G[u][v]['weight'] for u, v in edges] # 10 to magnify width
    nx.draw_networkx(G, pos, node_size=800, with_labels=True, node_color=node_colors,
                     edge_color=edge_colors, width=weights,
                     font_weight='bold')
    # edges
    plt.savefig('co_occurance_graph.png')
    plt.show()

def show_pcd(pcd, labels, full_tree=None, box=None):
    if box is None:
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pcd.get_axis_aligned_bounding_box())
    box.paint_uniform_color([.765, .129, .282])

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
        full_tree.colors = o3d.utility.Vector3dVector(np.ones((n_points,3))*[1/255,211/255,100/255]) #) [255/255,255/255,255/255]
        #vis.add_geometry(full_tree)
        pcd += full_tree
    vis.add_geometry(pcd)
    #vis.add_geometry(box)

    o3d.io.write_point_cloud('pcd_for_redering.ply', pcd)
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

def copy_with_scaling(label_from_path, label_to_path, img_from_path, img_to_path, scale_factor):
    if scale_factor == 1.:
        shutil.copy(label_from_path, label_to_path)
    else:
        im = cv2.imread(label_from_path)
        new_size = im.shape[:2]
        new_size = (int(new_size[1] * scale_factor), int(new_size[0] * scale_factor))

        im = cv2.resize(im, new_size,  cv2.INTER_NEAREST)
        cv2.imwrite(label_to_path, im)

        im = cv2.imread(img_from_path)
        im = cv2.resize(im, new_size, cv2.INTER_NEAREST)
        cv2.imwrite(img_to_path, im)

######################################################## Function to draw overly cand coping label images###############
def overly_mask_with_projection(projection_dir, label_dir, orig_img_dir, args):
    print("Generating overly and coping label frames")
    cam_dirs = [os.path.basename(x) for x in glob.glob(f"{projection_dir}/cam_*")]
    overlay_dir = os.path.join(projection_dir, 'overlay')
    os.makedirs(overlay_dir, exist_ok=True)
    for cam_dir in tqdm(cam_dirs):
        subdir = os.path.join(projection_dir, cam_dir)
        img_name = os.path.split(glob.glob(os.path.join(subdir,'frame*.png'))[0])[-1]
        segment_img_path = os.path.join(subdir, img_name)
        frame_label_path =  os.path.join(label_dir, f'label_{img_name}')
        copy_with_scaling(frame_label_path, os.path.join(subdir, f'label_{img_name}'),
                          os.path.join(orig_img_dir, img_name),  segment_img_path, args.scale_factor)

        # overlay projection on top of the segmentation image for debuging purpose
        projection_img_files = glob.glob(os.path.join(subdir, f'{args.visible_img_prefix}*.png'))
        seg_frame = cv2.imread(segment_img_path)
        # if args.scale_factor != 1.:
        #     seg_frame = cv2.resize(seg_frame, (
        #     int(seg_frame.shape[1] * args.scale_factor), int(seg_frame.shape[0] * args.scale_factor)))
        merged = np.zeros_like(seg_frame, dtype=np.uint8)

        for i in range(len(projection_img_files)):
            proj_img = cv2.imread(projection_img_files[i])
            mask = proj_img.astype(bool)
            merged[mask] = proj_img[mask]

        overlayed_img = cv2.addWeighted(seg_frame, .5, merged, .5, 0)
        cv2.imwrite(segment_img_path, overlayed_img)
        cv2.imwrite(os.path.join(overlay_dir, f'label_{img_name}'), overlayed_img)

#
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
#         projection_img_file = glob.glob(os.path.join(subdir, 'visible*.png'))[0]
#         seg_frame = cv2.imread(segment_img_path)
#         merged = np.zeros_like(seg_frame, dtype=np.uint8)
#
#         #for i in range(len(projection_img_files)):
#         proj_img = cv2.imread(projection_img_file)
#         mask = proj_img.astype(bool)
#         merged[mask] = proj_img[mask]
#
#         overlayed_img = cv2.addWeighted(seg_frame, .5, merged, .5, 0)
#         cv2.imwrite(segment_img_path, overlayed_img)
#         cv2.imwrite(os.path.join(overlay_dir, f'label_{img_name}'), overlayed_img)




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
                #reliability = vis_segmentaion_overlap_area / wo_occ_area
                #reliability =  vis_area/ wo_occ_area
                #reliability = vis_segmentaion_overlap_area/vis_area
                reliability = np.ones_like(wo_occ_area)

            # reliability += eps
            # reliability/=reliability.max()
            # reliability = np.clip(-np.log(1/reliability - 1)/20., 0, 1)
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
    #


# def infer_count(arg):


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--base_dir', type=str, required=False, default=r'C:\Users\MuzaddidMdAhmedAl\docker_mount')
    parser.add_argument('--base_dir', type=str, required=False, default=r'D:\3d_phenotyping') #D:\3d_phenotyping
    parser.add_argument('--recording_name', type=str, required=False, default='recording_2024-09-11_12-31-01') #
    parser.add_argument('--visible_img_prefix', type=str, required=False, default='visible_cluster')
    parser.add_argument('--wo_occ_img_prefix', type=str, required=False, default='wo_occ_cluster')

    parser.add_argument('--area_normalize', type=bool, required=False, default=False)
    parser.add_argument('--visualize', type=bool, required=False, default=True)
    parser.add_argument('--vis_full_tree', type=bool, required=False, default=True)
    parser.add_argument('--graph_partition', type=str, required=False, default='clique')#'community'

    parser.add_argument('--super_cluster_idx', type=int, required=False, default=-1)  # -1 to consider all
    parser.add_argument('--binary_threshold', type=int, required=False, default=100)
    parser.add_argument('--frame_sampling_interval', type=int, required=False, default=10)
    parser.add_argument('--n_thread', type=int, required=False, default=10)
    parser.add_argument('--scale_factor', type=float, required=False, default=1.)

    args = parser.parse_args()

    projection_dir = os.path.join(args.base_dir,'artifacts', args.recording_name, 'projection')
    pcd_dir = os.path.join(args.base_dir,'artifacts', args.recording_name, 'pcd')
    label_dir = os.path.join(args.base_dir,'training_data', args.recording_name, 'SegmentationLabel')
    orig_img_dir = os.path.join(args.base_dir,'training_data', args.recording_name, 'SegmentationObject')
    pcd_path = os.path.join(pcd_dir, 'all_super_cluster_info.npy' )
    pcd_data = np.load(pcd_path, allow_pickle=True)
    n_super_clusters = len(pcd_data)
    super_point_per_cluster = pcd_data[0]['aabb'].shape[0]


    def handle_single_super_cluster(super_cluster_idx):
        super_cluster_dir = os.path.join(projection_dir, f'super_cluster_{super_cluster_idx}')

        if not os.path.exists(os.path.join(super_cluster_dir, 'overlay')):
            overly_mask_with_projection(super_cluster_dir, label_dir, orig_img_dir, args)

        cluster_prop = process_super_cluster(super_cluster_dir,
                                             super_point_per_cluster,
                                             args.binary_threshold, args)
        affinity = calc_affinity(cluster_prop)
        # np.set_printoptions(precision=3, suppress=True)
        # print(affinity)

        '''Find the number of connected components'''
        #norm_affinity = affinity / np.abs(affinity.max(axis=1, keepdims=True)) ###**** buggy!!! handle case when max 0
        #norm_affinity = np.where(np.abs(norm_affinity) < .3, 0, norm_affinity)
        #affinity = np.where(affinity > 0, 1, 0)
        n_components, labels = get_component(affinity, args.graph_partition, draw=True)

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
        sc_ids = [i for i in range(min(17, n_super_clusters))]
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
            full_tree = full_tree.voxel_down_sample(voxel_size=7 * 10e-4)

        if len(affinity_list) ==1:
            draw_graph_from_adjacency_matrix(affinity_list[0])

        highlight_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            pcd.get_axis_aligned_bounding_box())
        pcd = show_pcd(pcd, np.concatenate(labels_list), None, None) #highlight_box
        if args.super_cluster_idx == -1:
            o3d.io.write_point_cloud(os.path.join(pcd_dir, "full_tree_seg_result.ply"), pcd)


if __name__ == '__main__':
    main()

