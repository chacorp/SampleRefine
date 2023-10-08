import os
import numpy as np
import cv2
import torch
import PIL.Image as Image
from scipy.interpolate import griddata
import argparse
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

def getSymXYcoordinates(iuv, resolution = 256, sym=False):
    xy, xyMask = getXYcoor(iuv, resolution = resolution)
    if sym:
        f_xy, f_xyMask = getXYcoor(flip_iuv(iuv), resolution = resolution)
        f_xyMask = np.clip(f_xyMask-xyMask, a_min=0, a_max=1)
        # combine actual + symmetric
        combined_texture = xy*np.expand_dims(xyMask,2) + f_xy*np.expand_dims(f_xyMask,2)
        combined_mask = np.clip(xyMask+f_xyMask, a_min=0, a_max=1)
        return combined_texture, combined_mask, f_xyMask
    else:
        texture = xy*np.expand_dims(xyMask,2)
        mask = np.clip(xyMask, a_min=0, a_max=1)
        return texture, mask, mask

def flip_iuv(iuv):
    POINT_LABEL_SYMMETRIES = [ 0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
    i = iuv[:,:,0]
    u = iuv[:,:,1]
    v = iuv[:,:,2]
    i_old = np.copy(i)
    for part in range(24):
        if (part + 1) in i_old:
            annot_indices_i = i_old == (part + 1)
            if POINT_LABEL_SYMMETRIES[part + 1] != part + 1:
                    i[annot_indices_i] = POINT_LABEL_SYMMETRIES[part + 1]
            if part == 22 or part == 23 or part == 2 or part == 3 : #head and hands
                    u[annot_indices_i] = 255-u[annot_indices_i]
            if part == 0 or part == 1: # torso
                    v[annot_indices_i] = 255-v[annot_indices_i]
    return np.stack([i,u,v],2)

def getXYcoor(iuv, resolution = 256):
    x, y, u, v = mapper(iuv, resolution)
    # A meshgrid of pixel coordinates
    nx, ny = resolution, resolution
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    ## get x,y coordinates
    uv_y = griddata((v, u), y, (Y, X), method='linear')
    uv_y_ = griddata((v, u), y, (Y, X), method='nearest')
    uv_y[np.isnan(uv_y)] = uv_y_[np.isnan(uv_y)]
    uv_x = griddata((v, u), x, (Y, X), method='linear')
    uv_x_ = griddata((v, u), x, (Y, X), method='nearest')
    uv_x[np.isnan(uv_x)] = uv_x_[np.isnan(uv_x)]
    # get mask
    uv_mask = np.zeros((ny,nx))
    uv_mask[np.ceil(v).astype(int),np.ceil(u).astype(int)]=1
    uv_mask[np.floor(v).astype(int),np.floor(u).astype(int)]=1
    uv_mask[np.ceil(v).astype(int),np.floor(u).astype(int)]=1
    uv_mask[np.floor(v).astype(int),np.ceil(u).astype(int)]=1
    kernel = np.ones((3,3),np.uint8)
    uv_mask_d = cv2.dilate(uv_mask,kernel,iterations = 1)
    # update
    coor_x = uv_x * uv_mask_d
    coor_y = uv_y * uv_mask_d
    coor_xy = np.stack([coor_x, coor_y], 2)
    return coor_xy, uv_mask_d

def mapper(iuv, resolution=256):
    dp_uv_lookup_256_np = np.load('utils/mapping/dp_uv_lookup_256.npy')
    H, W, _ = iuv.shape
    iuv_raw = iuv[iuv[:, :, 0] > 0]
    x = np.linspace(0, W-1, W).astype(np.int)
    y = np.linspace(0, H-1, H).astype(np.int)
    xx, yy = np.meshgrid(x, y)
    xx_rgb = xx[iuv[:, :, 0] > 0]
    yy_rgb = yy[iuv[:, :, 0] > 0]
    # modify i to start from 0... 0-23
    i = iuv_raw[:, 0] - 1
    u = iuv_raw[:, 1]
    v = iuv_raw[:, 2]
    uv_smpl = dp_uv_lookup_256_np[
        i.astype(np.int),
        u.astype(np.int),
        v.astype(np.int),
    ]
    u_f = uv_smpl[:, 0] * (resolution - 1)
    v_f = (1 - uv_smpl[:, 1]) * (resolution - 1)
    return xx_rgb, yy_rgb, u_f, v_f

def pad_PIL(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='data', help="path to image file to process. ex: ./train.lst")
    parser.add_argument("--save_path",  type=str, default='uv_data', help="path to save the uv data")
    parser.add_argument("--dp_path",    type=str, help="path to densepose data")
    parser.add_argument("--sym",        action='store_true', help="path to densepose data")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    from glob import glob
    images = sorted(glob(os.path.join(args.image_file, '*.png')))
    iuvs   = sorted(glob(os.path.join(args.dp_path, '*.png')))
    # for file in glob(os.path.join(args.image_file, '*.png')):
    #     if ('_iuv' in file) or ('_sil' in file):
    #         continue
    #     else:
    #         images.append(file)

    # import time
    idx = 0
    for im_name, iuv_name in zip(images,iuvs):
        # begin = time.time()

        im = Image.open(im_name)
        w, h = im.size

        # dp = os.path.join(im_name.split('.')[0]+'_iuv.png')

        # iuv = cv2.imread(iuv_name)
        iuv = np.array(Image.open(iuv_name))[..., ::-1]
        # iuv = iuv.transpose(1,0,2)

        iuv_h, iuv_w, _ = iuv.shape
        if np.sum(iuv[:,:,0]==0)==(iuv_h*iuv_w):
            raise ValueError('no human: invalid image %d: %s'%(idx, im_name))
        
        # print("[1] time for loading: {}".format(time.time()-begin))
        # start = time.time()
        # import pdb;pdb.set_trace()

        # uv_coor, uv_mask, uv_symm_mask  = getSymXYcoordinates(iuv, resolution=iuv_h, sym=args.sym)
        uv_coor, uv_mask, uv_symm_mask  = getSymXYcoordinates(iuv, resolution=512, sym=args.sym)
        # print("[2] time for getting symmetry coordinates: {}".format(time.time()-start))
        # start = time.time()

        # uv_coor_path = os.path.join(args.save_path, im_name.split('/')[-1].split('.')[0]+'_uv_coor.npy')
        # uv_coor = np.load(uv_coor_path)

        shift = int((h-w)/2)
        uv_coor[:,:,0] = uv_coor[:,:,0] + shift # put in center
        uv_coor = ((2*uv_coor/(h-1))-1)
        uv_coor = torch.from_numpy(uv_coor).float().unsqueeze(0) # [1, 512, 512, 2]

        x1 = shift
        x2 = h-(w+x1)
        im = pad_PIL(im, 0, x2, 0, x1, color=(0, 0, 0))

        im = torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(0).float()
        # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        # rgb_uv = torch.nn.functional.grid_sample(im.cuda(), uv_coor.cuda())
        rgb_uv = F.grid_sample(im, uv_coor, align_corners=True)
        rgb_uv = rgb_uv[0].permute(1,2,0).data.cpu().numpy() * np.expand_dims(uv_mask, 2)

        # rgb_uv = rgb_uv / 255

        # print("[3] time for grid sampling: {}".format(time.time()-start))
        # start = time.time()

        # ToPILImage()(rgb_uv[0]).save(os.path.join(args.save_path, im_name.split('/')[-1].split('.')[0]+'_rgb_uv_nosym.png'))
        save_path = os.path.join(args.save_path, im_name.split('/')[-1].split('.')[0]+'_rgb_uv_nosym.png')
        Image.fromarray(rgb_uv.astype(np.uint8)).save(save_path)
        idx = idx+1
        # print("[4] time for total: {}".format(time.time()-begin))
        break