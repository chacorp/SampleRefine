import random

import torch
import torchvision.transforms as T
import kornia.geometry.transform as kgt

from utils.constants import BOX, GRID
# Reference: https://github.com/cheind/py-thin-plate-spline

# TODO:
# 1.Get patches per body index
# 2.Apply transform in each patches
# 3.re-assemble transformed patches into image

def translate_transform(img, p=0.999, scale=0, t_range=0.1, input_size=256, label_mask=None, device='cpu'):
    """
    Params
        img    : (torch.Tensor) [..., H, W]
        p      : (float) probability
        t_range: (float) translation ratio
        s_range: (int)   Euler angle
        d_range: (int)   Euler angle
    """
    if random.random() > p:
        return img

    if scale > 0:
        t_range = scale * 0.4 # == 1 / 2.5

    # affine transform
    affine = T.RandomAffine(
        degrees   = 0,
        translate = (t_range, t_range),
    )
    # bound boxes for each input size
    denominator = 512 / input_size
    box = torch.ceil(BOX / denominator).type(torch.int32)
        
    # apply same affine transform (head, body leg, arm)
    for i in range(6):
        # box.shape: [4, 2] // box item = (left top, left bottom, right top, right bottom)
        l_top, r_bot = box[i]
        H_min, W_min, H_max, W_max = l_top[0], l_top[1], r_bot[0], r_bot[1]
        
        # crop patch
        patch_img = img[..., H_min:H_max, W_min:W_max]

        # apply affine
        warped_img = affine(patch_img)

        # replace patch
        img[..., H_min:H_max, W_min:W_max] = warped_img
    return img

def rotate_transform(img, p=0.999, scale=0, d_range=30, input_size=256, label_mask=None, device='cpu'):
    """
    Params
        img    : (torch.Tensor) [3, H, W]
        p      : (float) probability
        t_range: (float) translation ratio
        s_range: (int)   Euler angle
        d_range: (int)   Euler angle
    """
    if scale > 0:
        d_range = max(d_range * scale * 4, d_range) * 0.5

    if random.random() < p:
        # affine transform
        affine = T.RandomAffine(degrees = (-d_range, d_range))
        # bound boxes for each input size
        denominator = 512 / input_size
        box = torch.ceil(BOX / denominator).type(torch.int32)
            
        # apply same affine transform (head, body leg, arm)
        for i in range(6):
            # box.shape: [4, 2] // box item = (left top, left bottom, right top, right bottom)
            l_top, r_bot = box[i]
            H_min, W_min, H_max, W_max = l_top[0], l_top[1], r_bot[0], r_bot[1]
            
            # crop patch
            patch_img = img[..., H_min:H_max, W_min:W_max]

            # apply affine
            warped_img = affine(patch_img)

            # replace patch
            img[..., H_min:H_max, W_min:W_max] = warped_img
    return img

def sheer_transform(img, p=0.999, scale=0, s_range=30, input_size=256, label_mask=None, device='cpu'):
    """
    Params
        img    : (torch.Tensor) [3, H, W]
        p      : (float) probability
        t_range: (float) translation ratio
        s_range: (int)   Euler angle
        d_range: (int)   Euler angle
    """
    if scale > 0:
        s_range = max(s_range * scale * 4, s_range) * 0.5

    if random.random() < p:
        # affine transform
        affine = T.RandomAffine(
            degrees   = 0,
            shear     = (-s_range, s_range),
        )
        # bound boxes for each input size
        denominator = 512 / input_size
        box = torch.ceil(BOX / denominator).type(torch.int32)
            
        # apply same affine transform (head, body leg, arm)
        for i in range(6):
            # box.shape: [4, 2] // box item = (left top, left bottom, right top, right bottom)
            l_top, r_bot = box[i]
            H_min, W_min, H_max, W_max = l_top[0], l_top[1], r_bot[0], r_bot[1]
            
            # crop patch
            patch_img = img[..., H_min:H_max, W_min:W_max]

            # apply affine
            warped_img = affine(patch_img)

            # replace patch
            img[..., H_min:H_max, W_min:W_max] = warped_img
    return img

def affine_transform(img, p=0.999, scale=0, t_range=0.1, s_range=10, d_range=30, input_size=256, label_mask=None, device='cpu'):
    """
    Params
        img    : (torch.Tensor) [3, H, W]
        p      : (float) probability
        t_range: (float) translation ratio
        s_range: (int)   Euler angle
        d_range: (int)   Euler angle
    """
    if scale > 0:
        t_range = scale * 0.4 # == 1 / 2.5
        d_range = max(d_range * scale * 4, d_range) * 0.5
        s_range = d_range
        
    if random.random() < p:
        # affine transform
        affine = T.RandomAffine(
            degrees   = (-d_range, d_range), 
            translate = ( t_range, t_range),
            shear     = (-s_range, s_range),
        )
        # bound boxes for each input size
        denominator = 512 / input_size
        box = torch.ceil(BOX / denominator).type(torch.int32)
            
        # apply same affine transform (head, body leg, arm)
        for i in range(6):
            # box.shape: [4, 2] // box item = (left top, left bottom, right top, right bottom)
            l_top, r_bot = box[i]
            H_min, W_min, H_max, W_max = l_top[0], l_top[1], r_bot[0], r_bot[1]
            
            # crop patch
            patch_img = img[..., H_min:H_max, W_min:W_max]

            # apply affine
            warped_img = affine(patch_img)

            # replace patch
            img[..., H_min:H_max, W_min:W_max] = warped_img
    return img

def tps_transform_kornia(img, p=0.999, scale=0.1, vis=False, corner=True, input_size=256, label_mask=None, device='cpu'):
    """
    Params
        img  : (torch.Tensor) [B, 4, H, W] :: rgb texture + mask
        p    : (float) probability
        scale: (float)
        vis  : (bool)  visualize warping points for debug
        step : (int)   points in dimension
    """
    if random.random() > p:
        return img
    box = torch.ceil(BOX * input_size / 512).type(torch.int32)
    
    for i in range(6):            
        mask = label_mask[i]
        grid = GRID[i] # [3, 4] H, W
        bbox = box[i]
        (H_min, W_min), (H_max, W_max) = bbox[0], bbox[1]
        part_img = img * mask
        
        part_img = part_img[..., H_min:H_max, W_min:W_max]
        B, C, H, W = part_img.shape

        off_scale = scale / torch.max(grid[0], grid[1])

        c = torch.zeros(B, grid[0], grid[1], 2)
        c[..., 0] = torch.linspace(0, W, grid[1])
        c[..., 1] = torch.linspace(0, H, grid[0]).unsqueeze(-1)
        denom = max(W, H)
        c = (c / denom) * 2 - 1

        src_points = c.reshape(B, -1, 2).to(device)
        alpha      = torch.Tensor(src_points.shape).uniform_(-off_scale, off_scale).to(device)
        dst_points = src_points + alpha
        
        ### fix only corners
        if corner:
            dst_points[:,         0] = src_points[:,         0]                  # top left
            dst_points[:, grid[1]-1] = src_points[:, grid[1]-1]                  # top right
            dst_points[:,  -grid[0]] = src_points[:,  -grid[0]]                  # bottom left
            dst_points[:,        -1] = src_points[:,        -1]                  # bottom right
        ### fix all edges
        else: 
            dst_points[:, 0:grid[1]] = src_points[:, 0:grid[1]]                     # first row
            dst_points[:, -grid[1]:] = src_points[:, -grid[1]:]                     # last  row
            for s in range(grid[0]):
                dst_points[:, s*grid[1]] = src_points[:, s*grid[1]]                 # first col
                dst_points[:, ((s+1)*grid[1])-1] = src_points[:, ((s+1)*grid[1])-1] # last  col

        # using kornia TPS
        kernel_weights, affine_weights = kgt.get_tps_transform(src_points, dst_points)
        warped_image = kgt.warp_image_tps(part_img, src_points, kernel_weights, affine_weights, align_corners=True)
        
        if vis:
            src_points = norm_range(src_points) * (denom-1)
            dst_points = norm_range(dst_points) * (denom-1)
            
            warped_image[0][:3] = draw_keypoints(warped_image[0][:3], src_points, colors="red", radius=1)
            warped_image[0][:3] = draw_keypoints(warped_image[0][:3], dst_points, colors="blue", radius=1)

        img[..., H_min:H_max, W_min:W_max] = warped_image
    return img

def naive_tps_transform_kornia(img, p=0.999, scale=0.1, vis=False, corner=True, input_size=256, label_mask=None, device='cpu'):
    """
    Params
        img  : (torch.Tensor) [3, H, W]
        p    : (float) probability
        scale: (float)
        step : (int)   points in dimension
    """
    if random.random() > p:
        return img
    step = 6
    # C, H, W = img.shape
    B, C, H, W = img.shape

    off_scale = scale / step

    c = torch.zeros(B, step, step, 2)
    c[..., 0] = torch.linspace(0, W, step)
    c[..., 1] = torch.linspace(0, H, step).unsqueeze(-1)
    denom = max(W, H)
    c = (c / denom) * 2 - 1

    src_points = c.reshape(B, -1, 2).to(device)
    alpha      = torch.Tensor(src_points.shape).uniform_(-off_scale, off_scale).to(device)
    dst_points = src_points + alpha
    
    ### fix only corners
    if corner:
        dst_points[:,      0] = src_points[:,      0]                       # top left
        dst_points[:, step-1] = src_points[:, step-1]                       # top right
        dst_points[:,  -step] = src_points[:,  -step]                       # bottom left
        dst_points[:,     -1] = src_points[:,     -1]                       # bottom right
    ### fix all edges
    else: 
        dst_points[:, 0:step] = src_points[:, 0:step]                     # first row
        dst_points[:, -step:] = src_points[:, -step:]                     # last  row
        for s in range(step):
            dst_points[:, s*step] = src_points[:, s*step]                 # first col
            dst_points[:, ((s+1)*step)-1] = src_points[:, ((s+1)*step)-1] # last  col

    # using kornia TPS
    kernel_weights, affine_weights = kgt.get_tps_transform(src_points, dst_points)
    img = kgt.warp_image_tps(img, src_points, kernel_weights, affine_weights, align_corners=True)
    
    if vis:
        src_points = norm_range(src_points) * (denom-1)
        dst_points = norm_range(dst_points) * (denom-1)
        
        img[0][:3] = draw_keypoints(img[0][:3], src_points, colors="red", radius=1)
        img[0][:3] = draw_keypoints(img[0][:3], dst_points, colors="blue", radius=1)

    return img


def norm_range(img):
    if img.min() < 0:  # [-1 ~ 1] >>> [ 0 ~ 1]
        return (img + 1.0) * 0.5
    if img.min() >= 0: # [ 0 ~ 1] >>> [-1 ~ 1]
        return (img * 2.0) - 1.0
    
"""
    Modified draw_keypoints() from torchvision.utils
"""
from typing import List, Optional, Tuple, Union
@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """
    from PIL import Image, ImageDraw
    from torchvision.transforms import ToPILImage, ToTensor
    
    img_to_draw = ToPILImage()(image)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )
    return ToTensor()(img_to_draw)