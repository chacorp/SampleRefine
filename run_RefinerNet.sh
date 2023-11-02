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
    
CUDA_VISIBLE_DEVICES=${GPUS} python _inference.py \
	--input demo/demo_input/tex_1_angle_10005_symmetry.png \
	--pmask demo/demo_input/tex_1_angle_10005_mask_symm.png \
	--norm_map demo/demo_input/tex_1_angle_10005_norm.png \
 	--out infer \
	--G R1 \
	--num_blocks 9 \
	--num_layers 3 \
	--use_gate \
	--concat vis_mask_c \
    --Refine_mode blend \
 	--checkpoint R1-mirror-vis_mask_c \
    --which_epoch 11000 \
	--SamplerNet S1-mirror-norm_map_vis_mask