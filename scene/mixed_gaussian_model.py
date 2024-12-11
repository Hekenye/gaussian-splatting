import numpy as np
import os
import torch
import torch.nn as nn
import json

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


from .gaussian_model import GaussianModel
from utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_rotation, strip_symmetric, build_scaling_rotation)


class MixedGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
            self, 
            gaussians: GaussianModel, 
            head_mask: torch.Tensor, 
            optimizer_type="default",
    ):
        head_gaussians_attr = gaussians.capture_attr_with_mask(head_mask)
        (
            active_sh_degree,
            head_xyz, 
            head_feature_dc, 
            head_feature_rest,
            head_scaling,
            head_rotation,
            head_opacity,
        ) = head_gaussians_attr

        torso_gaussians_attr = gaussians.capture_attr_with_mask(~head_mask)
        (
            active_sh_degree,
            torso_xyz, 
            torso_feature_dc, 
            torso_feature_rest,
            torso_scaling,
            torso_rotation,
            torso_opacity,
        ) = torso_gaussians_attr

        self._head_xyz = nn.Parameter(head_xyz, requires_grad=True)
        self._head_features_dc = nn.Parameter(head_feature_dc, requires_grad=True)
        self._head_features_rest = nn.Parameter(head_feature_rest, requires_grad=True)
        self._head_scaling = nn.Parameter(head_scaling, requires_grad=True)
        self._head_rotation = nn.Parameter(head_rotation, requires_grad=True)
        self._head_opacity = nn.Parameter(head_opacity, requires_grad=True)

        self._torso_xyz = nn.Parameter(torso_xyz, requires_grad=True)
        self._torso_feature_dc = nn.Parameter(torso_feature_dc, requires_grad=True)
        self._torso_feature_rest = nn.Parameter(torso_feature_rest, requires_grad=True)
        self._torso_scaling = nn.Parameter(torso_scaling, requires_grad=True)
        self._torso_rotation = nn.Parameter(torso_rotation, requires_grad=True)
        self._torso_opacity = nn.Parameter(torso_opacity, requires_grad=True)

        self.active_sh_degree = active_sh_degree
        self.optimizer_type = optimizer_type

        self.setup_functions()

    @property
    def get_scaling(self):
        scaling = torch.cat([self._head_scaling, self._torso_scaling], dim=0)
        return self.scaling_activation(scaling)
    
    @property
    def get_rotation(self):
        rotation = torch.cat([self._head_rotation, self._torso_rotation], dim=0)
        return self.rotation_activation(rotation)
    
    @property
    def get_xyz(self):
        return torch.cat([self._head_xyz, self._torso_xyz], dim=0)
    
    @property
    def get_features(self):
        features_dc = torch.cat([self._head_features_dc, self._torso_feature_dc], dim=0)
        features_rest = torch.cat([self._head_features_rest, self._torso_feature_rest], dim=0)
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return torch.cat([self._head_features_dc, self._torso_feature_dc], dim=0)
    
    @property
    def get_features_rest(self):
        return torch.cat([self._head_features_rest, self._torso_feature_rest], dim=0)
    
    @property
    def get_opacity(self):
        opacity = torch.cat([self._head_opacity, self._torso_opacity], dim=0)
        return self.opacity_activation(opacity)
    
    def training_setup(self, training_args):
        l = [
            # {'params': [self._head_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._head_features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._head_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._head_opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._head_scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._head_rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(torch.cat([self._head_features_dc, self._torso_feature_dc], dim=0).shape[1]*torch.cat([self._head_features_dc, self._torso_feature_dc], dim=0).shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(torch.cat([self._head_features_rest, self._torso_feature_rest], dim=0).shape[1]*torch.cat([self._head_features_rest, self._torso_feature_rest], dim=0).shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(torch.cat([self._head_scaling, self._torso_scaling], dim=0).shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(torch.cat([self._head_rotation, self._torso_rotation], dim=0).shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = torch.cat([self._head_xyz, self._torso_xyz], dim=0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._head_features_dc, self._torso_feature_dc], dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = torch.cat([self._head_features_rest, self._torso_feature_rest], dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = torch.cat([self._head_opacity, self._torso_opacity], dim=0).detach().cpu().numpy()
        scale = torch.cat([self._head_scaling, self._torso_scaling], dim=0).detach().cpu().numpy()
        rotation = torch.cat([self._head_rotation, self._torso_rotation], dim=0).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_with_mask(self, path, mask):
        mkdir_p(os.path.dirname(path))

        xyz = torch.cat([self._head_xyz, self._torso_xyz], dim=0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._head_features_dc, self._torso_feature_dc], dim=0)[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = torch.cat([self._head_features_rest, self._torso_feature_rest], dim=0)[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = torch.cat([self._head_opacity, self._torso_opacity], dim=0)[mask].detach().cpu().numpy()
        scale = torch.cat([self._head_scaling, self._torso_scaling], dim=0)[mask].detach().cpu().numpy()
        rotation = torch.cat([self._head_rotation, self._torso_rotation], dim=0)[mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
