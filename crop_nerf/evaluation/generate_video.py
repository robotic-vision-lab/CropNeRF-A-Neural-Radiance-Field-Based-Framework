import open3d as o3d
import numpy as np
import cv2
import os
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

def pointcloud_to_video(pcd, labels=None, full_tree=None, output_path="output.mp4", num_frames=180, fps=20,  radius=1.0, height=.7, tilt_deg=10):
    # Load point cloud
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # if not pcd.has_points():
    #     raise ValueError("Point cloud is empty or could not be loaded.")
    #
    # # Create visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)  # offscreen rendering
    # vis.add_geometry(pcd)

    if labels is not None:
        max_label = labels.max()
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors = cmap(labels)  # / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.add_geometry(box)

    if full_tree:
        full_tree = full_tree.voxel_down_sample(voxel_size=5 * 10e-4)
        n_points = len(full_tree.points)
        full_tree.colors = o3d.utility.Vector3dVector(np.ones((n_points, 3)) * [155 / 255, 155 / 255, 155 / 255])
        #vis.add_geometry(full_tree)
        #pcd = pcd + full_tree
    vis.add_geometry(pcd)

    # Get render option (optional: adjust size, color, bg, etc.)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([1, 1, 1])  # white background
    render_opt.point_size = 2.0

    # # Camera params
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Get bounding box center (orbit target)
    center = pcd.get_center()
    tilt_rad = np.radians(tilt_deg)

    images = []
    for i in range(num_frames):
        angle = 2.0 * np.pi * i / num_frames

        # Camera moves in XY plane (circle) and is slightly above center
        cam_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height
        ])

        # Look slightly below the center for tilt effect
        target = center - np.array([0, 0, np.tan(tilt_rad) * radius])

        # Forward vector
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)

        # Global up vector (fix upside-down issue)
        up_world = np.array([0.0, 0.0, -1.0])

        # Right vector
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)

        # Recompute orthogonal up vector
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # Rotation matrix
        R_cam_to_world = np.stack([right, up, forward], axis=1)

        # Transformation matrix
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R_cam_to_world
        T_cam_world[:3, 3] = cam_pos

        # Inverse for extrinsic
        extrinsic = np.linalg.inv(T_cam_world).astype(np.float64)
        cam_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(cam_params)

        vis.poll_events()
        vis.update_renderer()

        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (255 * img).astype(np.uint8)
        images.append(img)

    vis.destroy_window()

    # Save as video
    h, w, _ = images[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img in images:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved orbit video to {output_path}")

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

    points = np.concatenate(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
    #pcd = pcd.crop(pcd_of_interest.get_axis_aligned_bounding_box())
    #show_pcd(pcd, np.asarray(labels), pcd_tree, None)
    pointcloud_to_video(pcd, np.asarray(labels), pcd_tree)

def generate_segmentation_output_vid(basedir, recording):
    input_path = os.path.join(basedir, recording, 'pcd', 'full_tree_seg_result.ply')
    pcd_data = o3d.io.read_point_cloud(input_path)
    pointcloud_to_video(pcd_data )

if __name__ == "__main__":
    basedir = r'D:\3d_phenotyping\artifacts'
    recording = 'recording_2024-09-11_12-31-01'

    #view_super_clusters(basedir, recording)
    generate_segmentation_output_vid(basedir, recording)
