import cv2
import numpy as np
import glob
import os
img_path = r"C:\Users\mxm6551xx\docker_mount_dir\recording_2024-09-11_12-48-18\SegmentationObject\frame_00001.png"
img_source_dir = r"C:\Users\mxm6551xx\docker_mount_dir\recording_2024-09-11_12-48-18\SegmentationObject"
img_write_dir = r"C:\Users\mxm6551xx\docker_mount_dir\recording_2024-09-11_12-48-18\SegmentationBoundary"
kernel = np.ones((19, 19), np.uint8)



def process(img_path):
    # read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_name = os.path.split(img_path)[1]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    unique_colors, n = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    for c in unique_colors[1:]:
        # Defining mask for detecting color
        ### ToDo: Optimize this line
        temp_mask = cv2.inRange(img, c, c)
        temp_mask =  cv2.morphologyEx(temp_mask, cv2.MORPH_GRADIENT, kernel)
        mask = cv2.add(mask, temp_mask)

    cv2.imwrite(os.path.join(img_write_dir, img_name), mask)

def convert_into_boundary_mask(img_source_dir):
    if not os.path.isdir(img_write_dir):
        os.mkdir(img_write_dir)
    for img_path in glob.glob(f"{img_source_dir}/*.png"):
        process(img_path)

convert_into_boundary_mask(img_source_dir)