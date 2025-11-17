import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

points = o3d.io.read_point_cloud("kitti_p.pcd")
pts = np.asarray(points.points)

app = gui.Application.instance
app.initialize()

vis = o3d.visualization.O3DVisualizer(" 394467238 ", 1024, 768)
vis.show_settings = True
vis.add_geometry("Points", points)
for idx in range(0, len(points.points)):
    if idx % 1000 == 0:
        vis.add_3d_label(points.points[idx], "{}".format(idx))

vis.reset_camera_to_default()

app.add_window(vis)
app.run()