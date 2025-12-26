import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path
import json
import cv2
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait

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
     #im /= -im[:, 2:3]
     return im


def get_super_cluster(super_cluster_info, idx):
    sup_cluster = super_cluster_info[idx]
    sup_pc = sup_cluster['pcd']
    cluster_pc = np.vstack([pc for _,pc in sup_pc.items()])
    return cluster_pc

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def get_frame_info(transform_file_path):
    meta = load_from_json(Path(transform_file_path))
    image_filenames = []
    c2w = []

    for frame in meta:
        filepath = Path(frame["file_path"])
        image_filenames.append(filepath)
        c2w.append(np.array(frame["transform"]))

    frames = {"image_filenames" : image_filenames,
              "c2w" : c2w}

    return frames

def update_buffer(z_buffer, pc, img, label, large=False):
    yx = pc[:,:2]/ -pc[:, 2:3]
    yx = np.round(yx).astype(int)
    ys = np.clip(yx[:,0], 0 , 1919)
    xs = np.clip(yx[:,1], 0,  1439)
    zs = -pc[:,2]
    if large:
        #update_idx = np.argwhere(z_buffer[xs, ys] > 1.1 * zs)
        img[xs, ys] = label
        z_buffer[xs, ys] = zs
        return z_buffer, img, (xs, ys)
    else:
        visible_xs, visible_ys = [], []
        for y, x, z in zip(ys,xs, zs):
            if z <=  z_buffer[x,y]:
                z_buffer[x,y] = z
                img[x,y] = label
                visible_xs.append(x)
                visible_ys.append(y)
        return z_buffer, img, (np.asarray(visible_xs, dtype=np.int32), np.asarray(visible_ys,dtype=np.int32))



def remove_part(full_tree, aabb):
    if isinstance(full_tree, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(full_tree)
    else:
        pcd = full_tree
    aabb = o3d.geometry.AxisAlignedBoundingBox( aabb[0], aabb[1])
    croped = pcd.crop(aabb)
    return np.asarray(croped.points)


def project_and_save_super_clusters(frames, cam_idx, cluster_data, full_tree_pc, full_semantic_pcd, save_dir):
    instance_mask_img = str(frames["image_filenames"][cam_idx].with_suffix(".png")).replace('images', 'SegmentationObject')
    z_buffer_init = np.ones((1440, 1920), dtype=np.float32) * np.inf
    img_init = np.zeros((1440, 1920), dtype=np.uint8)

    ## Get P from cam id
    c2w = frames["c2w"][cam_idx]
    P = get_projection_mat(fx, fy, cx, cy, c2w)
    proj = get_projection(P, full_tree_pc)
    t = time.time()
    z_buffer_init, img_init, _ = update_buffer(z_buffer_init, proj, img_init, label=0, large=True)
    #print("Projection took {:.3f} seconds.".format(time.time() - t))

    n_super_clusters = len(cluster_data)
    #t = time.time()
    for sup_idx in range(n_super_clusters):

        cam_dir = os.path.join(save_dir, f'super_cluster_{sup_idx}',  f'cam_{cam_idx}')
        if not os.path.exists(cam_dir): os.makedirs(cam_dir)
        sup_cluster = cluster_data[sup_idx]['pcd']
        z_buffer = z_buffer_init.copy()
        visible_label = img_init.copy()
        for sub_idx, pc in sup_cluster.items():
            aabb = cluster_data[sup_idx]['aabb'][sub_idx]
            pc = remove_part(full_semantic_pcd, aabb)
            # proj = get_projection(P, croped_tree_pc)
            # z_buffer, img, _ = update_buffer(z_buffer, proj, visible_label, label=0, large=True)
            # z_buffer = z_buffer_init.copy()
            # visible_label = img_init.copy()

            proj = get_projection(P, pc)
            z_buffer, visible_label, yx = update_buffer(z_buffer, proj, visible_label, sub_idx+1)

            # occlusion free projection
            occlusion_free_img = img_init.copy()
            occlusion_free_img[yx[0], yx[1] ] = 255
            write_fpath = os.path.join(cam_dir, f'occ_free_{sub_idx}.png')
            cv2.imwrite(write_fpath, occlusion_free_img)

            shutil.copy(instance_mask_img, cam_dir)
        # save projection as image
        write_fpath = os.path.join(cam_dir, f'visible_label.png')
        cv2.imwrite(write_fpath, visible_label)

        visible_img = np.where(visible_label > 0, 255, 0)
        write_fpath = os.path.join(cam_dir, f'visible.png')
        cv2.imwrite(write_fpath, visible_img)

    return time.time() - t
        #shutil.copy(segmentation_files[cam_idx], cam_dir)
        # calc overlap with the occlusion free
        # save it


base_dir = r'D:\3d_phenotyping\artifacts'
base_dir = r'/opt/data/artifacts'
recording = 'recording_2024-09-11_12-14-59'
input_path= os.path.join(base_dir, recording, 'pcd')

transform_file_path = os.path.join(base_dir, recording, 'transforms_train.json')
out_dir = os.path.join(base_dir, recording,'depth_projection')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
dataname = "semantics_pc.ply"
cluster_info_path = os.path.join(input_path, 'all_super_cluster_info.npy')
full_tree_path = os.path.join(input_path, 'full_tree_pc.ply')

frames = get_frame_info(transform_file_path)

full_tree_pcd =  o3d.io.read_point_cloud(full_tree_path)
full_tree_pc = np.asarray(full_tree_pcd.points)
full_semantic_pcd = o3d.io.read_point_cloud(os.path.join(input_path, 'semantics_pc.ply'))

cluster_info = np.load(cluster_info_path, allow_pickle=True)

n_frames = len(frames['c2w'])
t = time.time()
# with ThreadPoolExecutor(max_workers=10) as executor:
#     #outputs = list(executor.map(project_and_save_super_clusters, sc_ids))
#     futures = [executor.submit(project_and_save_super_clusters, frames, i, cluster_info, full_tree_pc, out_dir)
#                for i in range(n_frames)]
#     wait(futures)
# print(f'Task complete in:{time.time()-t}')
#
#for i in tqdm(range(n_frames)):
    #t = time.time()
project_and_save_super_clusters(frames, 13, cluster_info, full_tree_pc, full_semantic_pcd, out_dir)
    #print('time:', time.time() - t)

# plt.imshow(visible_img)
# # plt.scatter(u, v, s=0.1)
# plt.show()








