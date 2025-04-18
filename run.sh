#!/bin/bash

path_base=data
dset=bicycle
scene=train
path_source="${path_base}/${dset}"
output_base=output/exp_003
path_output="${output_base}/${dset}/${scene}"

# Model pre-training checkpoint (optional if you already have a pretrained model)
ckpt=output/exp_001_noquant/${dset}/${scene}/chkpnt15000.pth

ncls=4096
ncls_sh=512
ncls_dc=4096
kmeans_iters=5        # Can be adjusted to speed up training
kmeans_freq=200        # Can be increased to reduce the frequency of KMeans execution
st_iter=15000
max_iters=30000
max_prune_iter=20000
lambda_reg=1e-7
cuda_device=0
port=4060

CUDA_VISIBLE_DEVICES=$cuda_device python train_kmeans.py \
  --port $port \
  -s="$path_source" \
  -m="$path_output" \
  --kmeans_ncls "$ncls" \
  --kmeans_ncls_sh "$ncls_sh" \
  --kmeans_ncls_dc "$ncls_dc" \
  --kmeans_st_iter "$st_iter" \
  --kmeans_iters "$kmeans_iters" \
  --total_iterations "$max_iters" \
  --quant_params sh dc rot scale\
  --kmeans_freq 100 \
  --opacity_reg \
  --lambda_reg "$lambda_reg" \
  --max_prune_iter "$max_prune_iter" \
  --eval \
  --train_test_exp ""