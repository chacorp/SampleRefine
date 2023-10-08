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
CUDA_VISIBLE_DEVICES=${GPUS} python _test_pix2pix.py \
	--mode 'test' \
	--checkpoint Sampler4_-mirror-norm_map_vis_mask \
	--data_size 256 \
	--batch_size 8 \
	--which_epoch 30000 \
	--num_D 1 \
	--attention none \
	--num_blocks 8 \
	--num_layers 3 \
	--G Sampler4_ \
	--test_angle -1 \
	--mirror \
	--concat norm_map_vis_mask \
	--usermemo TEST