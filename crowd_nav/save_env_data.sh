#!/bin/bash


for ((i = 0; i < 100; i++));
    do
        echo "Test case $i"
        python test.py --policy tree-search-rl --model_dir data/360_HyperVnet_HHICM_embDim=from32to2_human10 --phase test --visualize --test_case $i --hyperbolic --video_file /home/aleflabo/amsterdam/intrinsic-rewards-navigation/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/video_submission/hyp2nav2/episode_ --embedding_dimension 2 --human_num 10

    done

# echo "Test case 16"
# mkdir -p data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/16
# python test.py --policy tree-search-rl --model_dir data/360_HyperVnet_HHICM_embDim=from32to2_human10 --phase test --visualize --test_case 16 --hyperbolic --video_file /home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/larger_than08/video_model_HHICM_embDIm=2_test_16 --embedding_dimension 2 --human_num 10

# python test.py --policy tree-search-rl --model_dir data/360_HyperVnet_HHICM_embDim=from32to2_human10 --phase test --visualize --test_case 35 --hyperbolic --video_file /home/aleflabo/amsterdam/intrinsic-rewards-navigation/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/correlation/video_model_HHICM_embDIm=2_test_ --embedding_dimension 2 --human_num 10
