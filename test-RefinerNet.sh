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
	--checkpoint Refiner-mirror-vis_mask_c \
	--data_size 256 \
	--batch_size 8 \
	--which_epoch 10000 \
	--num_D 1 \
	--attention none \
	--num_blocks 9 \
	--num_layers 3 \
	--G rTG \
	--test_angle -1 \
	--mirror \
	--concat vis_mask \
	--use_gate \
	--usermemo TEST