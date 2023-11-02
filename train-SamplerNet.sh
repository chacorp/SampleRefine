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
	--out S1-mirror-norm_map_vis_mask \
	--data_size 256 \
	--batch_size 8 \
	--epoch 30000 \
	--valid_iter 1000 \
	--niter 5000 \
	--niter_decay 50 \
	--lr 2e-4 --beta1 0.9 --beta2 0.999 \
	--lambda_L1 1 \
	--lambda_LPIPS 1 \
	--lambda_vgg 0 \
	--lambda_feat 0 \
	--lambda_gan 0 \
	--G S1 \
	--use_gate \
	--concat norm_map_vis_mask \
	--mirror \
	--masking \
	--progressive
