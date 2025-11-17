import open3d as o3d
import numpy as np
import imageio
import os

def render_video_from_camera_path(pcd_file, camera_json, output_video, width=1280, height=720, fps=30):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Load camera path JSON
    trajectory = o3d.io.read_pinhole_camera_trajectory(camera_json)

    # Create visualizer in headless mode
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    # Create video writer with explicit codec
    writer = imageio.get_writer(output_video, fps=fps, codec='libx264')

    for i, param in enumerate(trajectory.parameters):
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        writer.append_data(img)
        print(f"Captured frame {i+1}/{len(trajectory.parameters)}")

    vis.destroy_window()
    writer.close()
    print(f"Video saved to {output_video}")


if __name__ == "__main__":
    base_path = r'D:/OneDrive/PhD/RVL/code/3D_phenotyping/cotton_nerf/renders/recording_2024-09-11_12-42-36'
    render_video_from_camera_path(
        pcd_file=os.path.join(base_path, "pcd_for_rendering.ply"),
        camera_json=os.path.join(base_path, "2025-06-28-15-22-46.json"),
        output_video="rendered_video.mp4",
        width=1280,
        height=720,
        fps=30
    )
