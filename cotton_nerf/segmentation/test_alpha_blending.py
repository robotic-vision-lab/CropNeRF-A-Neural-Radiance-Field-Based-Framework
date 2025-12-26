import numpy as np
import cv2
import os

projection_dir = '/opt/data/projection/cam_1'
segment_img = cv2.imread(os.path.join(projection_dir, 'frame_00002.png' ))
#segment_img =  cv2.imread(os.path.join(projection_dir, 'frame_00002.jpg' ))
merged = np.zeros_like(segment_img)
for i in range(5):
    proj_img = cv2.imread(os.path.join(projection_dir, f'visible_cluster_{i}.png' ))

    mask = proj_img.astype(bool)
    merged[mask] = proj_img[mask]

segment_img = cv2.addWeighted(segment_img, .5, merged, .5, 0)

cv2.imwrite(os.path.join(projection_dir, f'dummpy.png'), segment_img)
