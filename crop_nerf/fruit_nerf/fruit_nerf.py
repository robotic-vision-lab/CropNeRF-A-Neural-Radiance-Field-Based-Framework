"""
FruitNeRF implementation .
"""

from __future__ import annotations
from collections import defaultdict
import shutil
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union, Optional
import os
import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import JaccardIndex

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.semantic_nerf_field import SemanticNerfField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    scale_gradients_by_distance_squared
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from nerfstudio.utils import colormaps
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox, SceneBox

from fruit_nerf.fruit_field import FruitField, SemanticNeRFField
from fruit_nerf.components.ray_samplers import UniformSamplerWithNoise

from segmentation import segmenter
from torchvision.utils import save_image
import cv2


@dataclass
class FruitNerfModelConfig(NerfactoModelConfig):
    """FruitModel Model Config"""

    _target: Type = field(default_factory=lambda: FruitModel)
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False
    num_layers_semantic: int = 2
    hidden_dim_semantics: int = 64
    geo_feat_dim: int = 15


class FruitModel(Model):
    """FruitModel based on Nerfacto model

    Args:
        config: FruitModel configuration to instantiate model
    """

    config: FruitNerfModelConfig

    def __init__(self, config: FruitNerfModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        self.test_mode = kwargs['test_mode']
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = FruitField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            num_layers_semantic=self.config.num_layers_semantic,
            hidden_dim_semantics=self.config.hidden_dim_semantics,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            geo_feat_dim=self.config.geo_feat_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_semantics=True,
            test_mode=self.test_mode,
            num_semantic_classes=1,
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        # Build the proposal network(s)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Samplers
        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def setup_inference(self, render_rgb, num_inference_samples):
        self.render_rgb = render_rgb  # True
        self.num_inference_samples = num_inference_samples  # int(200)
        self.proposal_sampler = UniformSamplerWithNoise(num_samples=self.num_inference_samples, single_jitter=False)
        self.field.spatial_distortion = None

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    # @torch.no_grad()
    # def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
    #     """Takes in a camera, generates the raybundle, and computes the output of the model.
    #     Assumes a ray-based model.
    #
    #     Args:
    #         camera: generates raybundle
    #     """
    #
    #     aabb_bounds = segmenter.get_bounds()
    #     aabb = SceneBox(torch.tensor([[-0.190317, -0.077369, -0.447199],[-0.0277847, 0.0316157, -0.346669]]))
    #     for i in range(len(aabb_bounds)):
    #         aabb = SceneBox(torch.tensor(aabb_bounds[i], dtype=torch.float32))
    #         outputs =  self.get_outputs_for_camera_ray_bundle(
    #             camera.generate_rays(camera_indices=0, keep_shape=True, aabb_box=aabb ) #aabb_box=aabb, obb_box=obb_box,
    #         )
    #
    #         save_image(outputs['rgb'].permute(2, 0, 1), f'test_pc_cloud_{i}.png' )
    #     return outputs

    @torch.no_grad()
    def get_outputs_for_projections(self,train_dataset, camera_optimizer = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """

        cameras = train_dataset.cameras
        segmentation_files = train_dataset.metadata['semantics'].filenames
        pcd_data = np.load("/opt/data/artifacts/pear/pcd/all_super_cluster_info_nsub_2.npy", allow_pickle=True)
        n_super_clusters = len(pcd_data)
        for i_sc in range(n_super_clusters):
            cluster_aabb = pcd_data[i_sc]['aabb']
            save_dir = f'/opt/data/artifacts/pear/projection/super_cluster_{i_sc}'
            os.makedirs(save_dir, exist_ok=True)
            #cluster_aabb = pcd_data.item().get('aabb')
            n_cluser = cluster_aabb.shape[0]



            for cam_idx, cam in enumerate(tqdm(cameras)):
                cam_dir = os.path.join(save_dir, f'cam_{cam_idx}')
                os.makedirs(cam_dir, exist_ok=True)
                #group_img =  torch.zeros(cameras[0].image_height, cameras[0].image_width, 3)

                for i in range(n_cluser):
                    aabb = SceneBox(torch.tensor(cluster_aabb[i], dtype=torch.float32))
                    rays = cam.generate_rays(camera_indices=0, keep_shape=True, aabb_box=aabb)

                    image_height, image_width = rays.origins.shape[:2]
                    valid_rays_mask = rays.nears < 1e10

                    n_valid_rays = valid_rays_mask.sum()
                    # Quick fix, model can not handle size 1 raybundle chunk
                    # if n_valid_rays % self.config.eval_num_rays_per_chunk == 1:
                    #     valid_rays_mask[valid_rays_mask][-1] = False

                    if valid_rays_mask.sum() < 10:
                        rgb_img = torch.zeros(image_height, image_width, 3)
                        save_image(rgb_img.permute(2, 0, 1), os.path.join(cam_dir, f'visible_cluster_{i}.png'))
                        save_image(rgb_img.permute(2, 0, 1), os.path.join(cam_dir, f'wo_occ_cluster_{i}.png'))
                        continue

                    # Generate unobstructed projection
                    rgb_img = torch.zeros(image_height * image_width, 3)
                    outputs = self.get_outputs_for_camera_jagged_ray_bundle(rays[valid_rays_mask.squeeze()])
                    rgb_img[valid_rays_mask.reshape(-1)] = outputs['semantics']
                    rgb_img = rgb_img.reshape(image_height, image_width, 3)
                    save_image(rgb_img.permute(2, 0, 1), os.path.join(cam_dir, f'wo_occ_cluster_{i}.png'))

                    # Generate projection of the visible parts
                    rays.fars[valid_rays_mask] = rays.nears[valid_rays_mask]
                    rays.nears[valid_rays_mask] = 0.0
                    weights = torch.zeros(image_height*image_width)
                    weights[valid_rays_mask.reshape(-1)] = self.get_density_for_camera_ray_bundle(rays[valid_rays_mask.squeeze()])
                    occlusion_mark = weights >= .5
                    occlusion_mark = occlusion_mark.reshape(image_height, image_width)
                    rgb_img[occlusion_mark] = 0.0

                    save_image(rgb_img.permute(2, 0, 1), os.path.join(cam_dir, f'visible_cluster_{i}.png'))
                shutil.copy(segmentation_files[cam_idx], cam_dir)

        return

    def get_density_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """

        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        num_rays = len(camera_ray_bundle)
        weight_list = []
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            #self.camera_optimizer.apply_to_raybundle(ray_bundle)
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle,
                                                                                density_fns=self.density_fns)

            field_outputs = self.field.forward(ray_samples)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weight_list.append(weights.cpu().squeeze().sum(-1))
        weights = torch.cat(weight_list)#.view(image_height, image_width, -1)
        return weights

    def get_outputs_for_camera_jagged_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        #image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)

            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output.cpu())
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list) # type: ignore
        #outputs['depth'] = outputs['depth'].to(torch.device("cuda"))
        return outputs

    # @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output.cpu())
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        outputs['depth'] = outputs['depth'].to(torch.device("cuda"))
        return outputs

    def semantic_projection(field_outputs, weight) -> torch.Tensor:
        outputs = {}
        semantic = field_outputs[FieldHeadNames.SEMANTICS][..., 0].reshape((-1, 1)).repeat((1, 3))
        density = field_outputs[FieldHeadNames.DENSITY][..., 0].reshape((-1, 1)).repeat((1, 3))

        semantic_labels = torch.sigmoid(semantic)
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)

        semantics_colormap = semantic_labels

        mask_sem = semantic >= 3  # 20
        mask_den = density >= 70  # 10
        mask_sem_colormap = semantics_colormap >= 0.99

        density_weight = weight.reshape(-1)
        mask_sem_colormap = mask_sem_colormap.sum(dim=1).to(bool) & mask_den.sum(dim=1).to(bool)
        semantic_weight = torch.where(mask_sem_colormap, density_weight, weight.min())
        return semantic_weight.reshape_as(weight)

    def get_projection_outputs(self, ray_bundle: RayBundle):
        outputs = {}

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        #depth = outputs[depth_output_name]
        points = ray_samples.frustums.get_positions()
        #point = ray_bundle.origins + ray_bundle.directions * depth
        #view_direction = ray_bundle.directions

        pcd_of_interest, _ = segmenter.get_dummy_pcd()
        pcd_of_interest = np.asarray(pcd_of_interest.points)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb,
                   "accumulation": accumulation,
                   "depth": depth,
                   "weights_list": weights_list,
                   "ray_samples_list": ray_samples_list}

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # semantics
        semantic_weights = weights
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels].repeat(1, 3)

        return outputs


    def get_export_outputs(self, ray_bundle: RayBundle):
        outputs = {}

        ray_samples = self.proposal_sampler(ray_bundle)
        field_outputs = self.field.forward(ray_samples)

        outputs["rgb"] = field_outputs[FieldHeadNames.RGB]

        outputs['point_location'] = ray_samples.frustums.get_positions()
        outputs["semantics"] = field_outputs[FieldHeadNames.SEMANTICS][..., 0]
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY][..., 0]

        semantic_labels = torch.sigmoid(outputs["semantics"])
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)

        outputs["semantics_colormap"] = semantic_labels

        return outputs


    def get_inference_outputs(self, ray_bundle: RayBundle):
        outputs = {}


        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb,
                   "accumulation": accumulation,
                   "depth": depth,
                   "weights_list": weights_list,
                   "ray_samples_list": ray_samples_list}

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # semantics
        semantic_weights = weights
        # masked_density = self.semantic_projection(field_outputs, field_outputs[FieldHeadNames.DENSITY])
        # semantic_weights = ray_samples.get_weights(masked_density)
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels].repeat(1, 3)

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):  #

        # apply the camera optimizer pose tweaks
        #if self.training:
        self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        #expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # semantics
        semantic_weights = weights
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels].repeat(1, 3)

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image[:, :3], outputs["rgb"])

        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
            outputs["semantics"], batch["fruit_mask"]
        )
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        # Add loss from camera optimizer
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        if self.test_mode == 'inference':
            # fruit_nerf_output = self.get_inference_outputs(ray_bundle, self.render_rgb)
            fruit_nerf_output = self.get_inference_outputs(ray_bundle)
        elif self.test_mode == 'export':
            # fruit_nerf_output = self.get_inference_outputs(ray_bundle, self.render_rgb)
            fruit_nerf_output = self.get_export_outputs(ray_bundle)
        else:
            fruit_nerf_output = self.get_outputs(ray_bundle)

        return fruit_nerf_output

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image[:,:3]) ############# image
        metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        rgb = torch.clamp(rgb, min=0, max=1)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        # semantics
        # semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        semantic_labels = torch.sigmoid(outputs["semantics"])
        images_dict[
            "semantics_colormap"] = semantic_labels

        # valid mask
        images_dict["fruit_mask"] = batch["fruit_mask"].repeat(1, 1, 3).to(self.device)
        # batch["fruit_mask"][batch["fruit_mask"] < 0.1] = 0
        # batch["fruit_mask"][batch["fruit_mask"] >= 0.1] = 1

        from torchmetrics.classification import BinaryJaccardIndex
        metric = BinaryJaccardIndex().to(self.device)
        semantic_labels = torch.nn.functional.softmax(outputs["semantics"])
        iou = metric(semantic_labels[..., 0], batch["fruit_mask"][..., 0])
        metrics_dict["iou"] = float(iou)

        return metrics_dict, images_dict


class FruitModelMLP(Model):
    pass
