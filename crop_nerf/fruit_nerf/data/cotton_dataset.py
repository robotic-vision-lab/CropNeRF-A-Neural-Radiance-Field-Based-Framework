# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic dataset.
"""

from typing import Dict

import torch
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

import cv2
import numpy

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

def get_object_semantics(pil_img):
    if pil_img.mode == "RGB":
        pil_img = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, semantics = cv2.threshold(numpy.array(pil_img), 3, 255, cv2.THRESH_BINARY)
    semantics = torch.from_numpy(np.float16(semantics))
    return semantics

def get_boundary_semantics(pil_img):
    kernel = np.ones((9, 9), np.uint8)
    img = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGB2BGR)
    unique_colors, n = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    semantics = np.zeros(img.shape[:2], dtype=np.uint8)

    for c in unique_colors:
        if not np.any(c): # check for [0, 0, 0] color which is the background
            continue
        # Defining mask for detecting color
        ### ToDo: Optimize this line
        temp_mask = cv2.inRange(img, c, c)
        thick_contour = cv2.morphologyEx(temp_mask,  cv2.MORPH_GRADIENT,  kernel)
        semantics = cv2.add(semantics, thick_contour)
    semantics = torch.from_numpy(np.float32(semantics))
    return semantics

def get_semantics_and_mask_tensors_from_path(
        filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    # semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    #semantics = get_boundary_semantics(pil_image)
    semantics = get_object_semantics(pil_image)
    if semantics.max() > 1.:
        semantics = semantics / 255
    else:
        raise ValueError("Please look at mask file manually! How to normalize")

    return semantics


class FruitDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask", "semantics"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics), "No semantic instance could be found! Is a semantic folder included in the input folder and transform.json file?"
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )

        return {"fruit_mask": semantic_label[..., None]}

    def get_image_float16(self, image_idx: int) :
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float16") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                    self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_data(self, image_idx: int, image_type = "float16") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        if image_type == "float16":
            image = self.get_image_float16(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
