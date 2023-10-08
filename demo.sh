# #!/bin/bash
# while [[ $# -gt 0 ]]; do
#   case $1 in
#     -g|--gpu)
#       GPUS="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     -p|--path)
#       PORT="$2"
#       shift # past argument
#       shift # past value
#     ;;
#     -*|--*)
#       echo "Unknown option $1"
#       exit 1
#       ;;
#     *)
#   esac
# done
# CUDA_VISIBLE_DEVICES=${GPUS} python _inference.py \
python _inference.py \
	--input demo/demo_input/tex_1_angle_10005_symmetry.png \
	--pmask demo/demo_input/tex_1_angle_10005_mask_symm.png \
	--norm_map demo/demo_input/tex_1_angle_10005_norm.png \
 	--out infer \
 	--checkpoint Refiner-mirror-vis_mask_c \
 	--usermemo Inference --infer