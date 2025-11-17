import pdb
import traceback
import open3d  # 0.16.0
import numpy as np

print('\n =============================== ')
print(f' ======  Open3D=={open3d.__version__}  ======= ')
print(' =============================== \n')

if __name__ == "__main__":

    try:

        # Step 0 - Init
        WIDTH = 1280
        HEIGHT = 720

        # Step 1 - Get scene objects
        meshFrame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        sphere1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere1.translate([0.1, 0.2, -5])

        # Step 2 - Create visualizer object
        vizualizer = open3d.visualization.Visualizer()
        vizualizer.create_window()
        vizualizer.create_window(width=WIDTH, height=HEIGHT)

        # Step 3 - Add objects to visualizer
        vizualizer.add_geometry(sphere1)
        vizualizer.add_geometry(meshFrame)

        # Step 4 - Get camera lines
        standardCameraParametersObj = vizualizer.get_view_control().convert_to_pinhole_camera_parameters()
        # cameraLines = open3d.geometry.LineSet.create_camera_visualization(intrinsic=standardCameraParametersObj.intrinsic, extrinsic=standardCameraParametersObj.extrinsic)
        cameraLines = open3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT,
                                                                          intrinsic=standardCameraParametersObj.intrinsic.intrinsic_matrix,
                                                                          extrinsic=standardCameraParametersObj.extrinsic)
        vizualizer.add_geometry(cameraLines)

        # Step 5 - Run visualizer
        vizualizer.run()

        # step 99 -- Debug
        pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()