#!/bin/bash
while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpu)
      GPUS="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
  esac
done
CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
	--out R1-mirror-vis_mask_c \
	--data_size 256 \
	--batch_size 4 \
	--epoch 30000 \
	--valid_iter 1000 \
	--niter 10000 \
	--niter_decay 50 \
	--lr 2e-4 --beta1 0.9 --beta2 0.999 \
	--lambda_L1 10 \
	--lambda_vgg 10 \
	--lambda_feat 1 \
	--lambda_gan 1 \
	--lambda_LPIPS 0 \
	--G R1 \
	--num_blocks 9 \
	--num_layers 3 \
	--use_gate \
	--concat vis_mask_c \
	--Refine_mode blend \
	--mirror \
	--aug_color \
	--p 0.4 \
	--SamplerNet S1-mirror-norm_map_vis_mask