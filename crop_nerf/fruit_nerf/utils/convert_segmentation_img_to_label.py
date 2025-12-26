import os
import numpy as np
import cv2
from tqdm import tqdm
import sys


def convert_img_to_label(recording, scale_factor=1):
    base_dir = '/opt/data' #r'D:\3d_phenotyping'
    #recording = 'tree_01'
    segmentation_object_dir = os.path.join(base_dir, f'training_data/{recording}/SegmentationObject')
    output_dir = os.path.join(base_dir, f'training_data/{recording}/SegmentationLabel')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    background_color = (0,0,0)
    frames = os.listdir(segmentation_object_dir)
    for frame in tqdm(frames):
        img = cv2.imread(os.path.join(segmentation_object_dir, frame))
        #pil_image.resize(newsize, resample=Image.NEAREST)
        if scale_factor != 1:
            new_size = (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)

        unique_col = np.unique(img.reshape(-1, 3), axis=0)
        unique_col = set([tuple(unique_col[i]) for i in range(len(unique_col))])
        unique_col.remove(background_color)
        color2label = {tuple(x): i for i, x in enumerate(unique_col, start=1)}
        color2label[background_color] = 0 # make sure backgroud is always 0

        img_shape = img.shape
        img = img.reshape(-1, 3)
        label_img = np.zeros((img_shape[0]* img_shape[1]), dtype=np.uint8)
        for i in range(img.shape[0]):
            label_img[i] = color2label[tuple(img[i])]
        write_fpath = os.path.join(output_dir, f'label_{frame}')
        cv2.imwrite(write_fpath, label_img.reshape(img_shape[0], img_shape[1]))


if __name__ == '__main__':

    convert_img_to_label(sys.argv[1])

       