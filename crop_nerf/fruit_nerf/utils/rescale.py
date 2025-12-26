import numpy as np
import cv2
import os
from glob import glob

def copy_with_scaling(img_from_path, img_to_path,  label_from_path, label_to_path, scale_factor=.25):

    if os.path.exists(img_from_path):
        im = cv2.imread(img_from_path)
        new_size = im.shape[:2]
        new_size = (int(new_size[1] * scale_factor), int(new_size[0] * scale_factor))
        im = cv2.resize(im, new_size, cv2.INTER_CUBIC)
        cv2.imwrite(img_to_path, im)
    else:
        return

    if os.path.exists(label_from_path):
        im = cv2.imread(label_from_path)
        im = cv2.resize(im, new_size,  cv2.INTER_NEAREST)
        cv2.imwrite(label_to_path, im)

def rescale(scale_factor=.25):
    source_img_dir = r'D:\3d_phenotyping\artifacts\tree_01\projection\super_cluster_0'
    for cam_dir in os.listdir(source_img_dir):
        cam_dir = os.path.join(source_img_dir, cam_dir)
        target_img_file = glob(os.path.join(cam_dir, 'frame_*'))
        if len(target_img_file) != 1:
            print("Zero or more frames found in {}".format(cam_dir))
        im = cv2.imread(target_img_file[0])
        new_size = im.shape[:2]
        new_size = (int(new_size[1] * scale_factor), int(new_size[0] * scale_factor))
        im = cv2.resize(im, new_size, cv2.INTER_NEAREST)
        cv2.imwrite(target_img_file[0], im)


    save_img_dir = r'D:\3d_phenotyping\training_data\tree_01\images_scaled'

def scale_label():
    source_img_dir = r'D:\3d_phenotyping\training_data\tree_01\images_orig'
    save_img_dir = r'D:\3d_phenotyping\training_data\tree_01\images_scaled'
    os.makedirs(save_img_dir, exist_ok=True)

    source_semantic_dir = r'D:\3d_phenotyping\training_data\tree_01\SegmentationObject_orig'
    save_semantic_dir = r'D:\3d_phenotyping\training_data\tree_01\SegmentationObject_scaled'
    os.makedirs(save_semantic_dir, exist_ok=True)

    for img_file in os.listdir(source_img_dir):
        img_name = os.path.split(img_file)[1]
        label_img_name = img_name.split('.')[0] + '.png'
        copy_with_scaling(os.path.join(source_img_dir, img_file), os.path.join(save_img_dir, img_name),
                          os.path.join(source_semantic_dir, label_img_name),
                          os.path.join(save_semantic_dir, label_img_name))


if __name__ == "__main__":
   rescale()