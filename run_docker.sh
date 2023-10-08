#!/bin/bash
while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpu)
      GPUS="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--port)
      PORT="$2"
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

docker run --gpus '"device='${GPUS}'"' \
    -it --rm --shm-size=8gb \
    -p ${PORT}:${PORT} \
    seokg1023/vml-pytorch:vessl