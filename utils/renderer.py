from glob import glob

import numpy as np
import torch
import torch.nn as nn

from pytorch3d.io import load_obj 
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams,
    TexturesUV,
)

from pytorch3d.transforms.rotation_conversions import _axis_angle_rotation

class Renderer(nn.Module):
    def __init__(self,
            device     = None,
            extend     = 1,
            elev       = (0.0), 
            azim       = (0.0), 
            image_size = 512, 
            batch_size = 1,
            distance   = 1,
            at_change  = True,
            at_xyz     = (0.0, 0.9, 0.0), 
            out_dir    = None,
            opt        = None,
        ):
        super().__init__()

        self.device         = device
        self.extend         = extend
        self.elev           = elev
        self.azim           = azim
        self.opt            = opt
        self.batch_size     = batch_size
        self.rot_mat        = None

        if self.opt:
            image_size      = self.opt.data_size
            out_dir         = self.opt.out
            distance        = self.opt.render_distance
            self.batch_size = self.opt.batch_size
            self.elev       = self.opt.render_elev
            self.azim       = self.opt.render_azim
            at_xyz          = self.opt.render_at
            scale_xyz       = ((self.opt.render_scale,) * 3,)
            if self.opt.mode == 'debug':
                import time
                self.time = time

        if len(self.elev) == 1:
            self.elev = self.elev[0]
        if len(self.azim) == 1:
            self.azim = self.azim[0]

        # Initialize a camera: 
        # the number of different viewpoints from which we want to render the mesh. (+X: left, +Y: up, +Z: in)
        R, T = look_at_view_transform(
            dist            = distance, 
            elev            = self.elev * self.batch_size, # fake and real
            azim            = self.azim * self.batch_size, # fake and real
            up              = ((0.0, 1.0, 0.0),),
            at              = (at_xyz,)
            # at            = ((0.0, 0.9, 0.0),) if at_change else ((0.0, 0.1, 0.0),) ,
        )

        select_camera = FoVOrthographicCameras(
            scale_xyz       = scale_xyz,
            R               = R, 
            T               = T,
            device          = device
        )

        blend_params = BlendParams(
            sigma           = 1e-8,
            gamma           = 1e-8,
            background_color= [0., 0., 0.]
        )

        # Rasterization settings for differentiable rendering, where the blur_radius initialization is based on Liu et al, 
        # 'Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning', ICCV 2019
        raster_settings = RasterizationSettings(
            image_size      = image_size, 
            blur_radius     = np.log(1. / 1e-8 - 1.) * blend_params.sigma, 
            faces_per_pixel = 10
        )

        lights = DirectionalLights(
            ambient_color   = [[1., 1., 1.]], 
            diffuse_color   = [[0., 0., 0.]],  
            specular_color  = [[0., 0., 0.]], 
            direction       = T, 
            device          = device
        )

        custom_rasterizer = MeshRasterizer(
            cameras         = select_camera, 
            raster_settings = raster_settings
        )

        custom_shader = SoftPhongShader(
            device          = device,
            cameras         = select_camera,
            lights          = lights,
            blend_params    = blend_params
        )

        renderer = MeshRenderer(
            rasterizer      = custom_rasterizer,
            shader          = custom_shader
        )

        self.renderer       = renderer
        self.camera         = select_camera
        self.lights         = lights

        self.out_dir        = out_dir
        if not self.out_dir:
            print('No output directory specified: self.out_dir == None, rendered image will not be saved')

        self.objs           = np.array(sorted(glob("../posed_obj_test/*.obj"))) # 50
        self.len_objs       = len(self.objs)
        self.load_objs_to_list(self.objs)

    def load_objs_to_list(self, objs):
        self.v_list  = []

        for i, f_obj in enumerate(objs):
            verts, faces, aux = load_obj(f_obj)
            
            if i == 0:
                self.f_list  = [faces.verts_idx.cuda()] * self.batch_size       # fake and real
                self.vt_list = [aux.verts_uvs.cuda()] * self.batch_size         # fake and real
                self.ft_list = [faces.textures_idx.cuda()] * self.batch_size    # fake and real
            self.v_list.append(verts.cuda())
        
    def load_meshes_with_textures(self, textures, random_list=None):
        """https://pytorch3d.readthedocs.io/en/latest/modules/structures.html"""
        if random_list == None:
            random_list     = torch.randperm(self.len_objs)[:self.batch_size]   # fake and real

        texturesUV          = TexturesUV(
                maps      = textures.permute(0,2,3,1),                          # B C H W -> B H W C
                faces_uvs = self.ft_list,
                verts_uvs = self.vt_list
            )
        meshes              = Meshes(
                verts     = [self.v_list[j] for j in random_list],              # only list
                faces     = self.f_list,
                textures  = texturesUV
            )
        return meshes, random_list

    def degree2radian(self, degree):
        return torch.deg2rad(torch.tensor([degree]))

    def random_rotation_Y(self):
        # rand_ang = torch.randint(-9, 10, (1,)) * 10
        # rand_rad = self.degree2radian(rand_ang)
        rand_rad = torch.rand(1) * torch.pi #* 2
        rot_mat  = _axis_angle_rotation("Y", rand_rad).squeeze(0)
        return rot_mat.to(self.device)

    def apply_random_rotation_Y(self, meshes):
        if self.rot_mat == None:
            # for fake texture
            self.rot_mat = self.random_rotation_Y()
            for B in range(0, self.batch_size):
                meshes._verts_list[B] = meshes._verts_list[B] @ self.rot_mat
        else:
            # for real texture
            for B in range(0, self.batch_size):
                meshes._verts_list[B] = meshes._verts_list[B] @ self.rot_mat
            self.rot_mat = None
        return meshes

    def get_R_T(self, azim, device):
        R, T = look_at_view_transform(
                    dist   =  1, 
                    elev   =  0.0,  # elevation
                    azim   =  azim, # azimuth
                    # at     =  ((0.0, 0.9, 0.0),), ### render
                    at     =  ((0.0, 0.0, 0.0),), #################### normal
                    up     =  ((0.0, 1.0, 0.0),),
                    device = device
                )
        return R, T

    def forward(self, textures, R=None, T=None, random_list=None):
        ## random shuffle textures in batch -> is shuffle necessary?
        meshes, random_list = self.load_meshes_with_textures(textures, random_list)

        ## random rotate on Y axis -> not used
        # meshes = self.apply_random_rotation_Y(meshes)

        ## if self.extend > 1:
        #     meshes = meshes.extend(self.extend)                                   # [B, C, ...] -> [B * self.azim, C, ...]
        
        if R!=None and T!=None:
            rendered = self.renderer(meshes, cameras=self.camera, lights=self.lights, R=R, T=T) # [B, H, W, C]
        else:
            rendered = self.renderer(meshes, cameras=self.camera, lights=self.lights) # [B, H, W, C]

        rendered = rendered.permute(0,3,1,2) # [B, C, H, W]
        return rendered, random_list
        # from torchvision.transforms import ToPILImage; ToPILImage()(rendered[3][:3]).save('vis_rendered.png')