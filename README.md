

# Hyp²Nav: Hyperbolic Planning and Curiosity for Crowd Navigation (IROS 2024)

_Guido D'Amely*, Alessandro Flaborea*, Pascal Mettes, Fabio Galasso_


The official PyTorch implementation of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024 paper [**Hyp²Nav: Hyperbolic Planning and Curiosity for Crowd Navigation**](https://arxiv.org/abs/2407.13567).


![Watch the video](video/iros_video.gif)

## Abstract
Autonomous robots are increasingly becoming a strong fixture in social environments. Effective crowd navigation requires not only safe yet fast planning, but should also enable interpretability and computational efficiency for working in real-time on embedded devices. In this work, we advocate for hyperbolic learning to enable crowd navigation and we introduce Hyp2Nav. Different from conventional reinforcement learning-based crowd navigation methods, Hyp2Nav leverages the intrinsic properties of hyperbolic geometry to better encode the hierarchical nature of decision-making processes in navigation tasks. We propose a hyperbolic policy model and a hyperbolic curiosity module that results in effective social navigation, best success rates, and returns across multiple simulation settings, using up to 6 times fewer parameters than competitor state-of-the-art models. With our approach, it becomes even possible to obtain policies that work in 2-dimensional embedding spaces, opening up new possibilities for low-resource crowd navigation and model interpretability. Insightfully, the internal hyperbolic representation of Hyp2Nav correlates with how much attention the robot pays to the surrounding crowds, e.g. due to multiple people occluding its pathway or to a few of them showing colliding plans, rather than to its own planned route.

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
- clone Python-RVO2 and cd in the repo
- `sudo apt-get install cmake`
- `cmake build .`
  - if this step does not work:
    - `pip install cmake`
- `pip install -r requirements.txt`
  - for python>=3.7: `pip install Cython`
- python setup.py build
  - if this step does not work:
    - `cd ..`
    - `rm -r Python-RVO2`
    - `git clone git@github.com:sybrenstuvel/Python-RVO2.git` (the Python-RVO2 repo)
- python setup.py install
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
- pip install 'socialforce[test,plot]'
3. Install crowd_sim and crowd_nav into pip
`pip install -e .`

## Getting Started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found [here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.

1. Train a policy.
```
python train.py --policy tree-search-rl --output_dir data/model_name/ --config configs/icra_benchmark/ts_HVNet_Hypercuriosity.py --gpu --wandb_mode online --wandb_name name_of_the_run --embedding_dimension=[2,128] --hyperbolic
```

2. Test policies with 1000 test cases.
```
python test.py --model_dir data/model_name --hyperbolic --embedding_dimension [2,64,128] --gpu --device cuda:0
```

3. Run policy for one episode and visualize the result.
```
python test.py --policy tree-search-rl --model_dir data/model_name --phase test --visualize --test_case [1,..,1000] --hyperbolic --video_file /path/to/video/file_path/ --embedding_dimension [2,128] --human_num [5,10]
```

Note that in **run_experiments_icra.sh**, some examples of how to train different policies with several exploration algorithms. In **configs/icra_benchmark/**, all the configurations used for testing are shown.



## Acknowledge
This work is based on [CrowdNav](https://github.com/vita-epfl/CrowdNav) and [RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning) and [SG-D3QN](https://github.com/nubot-nudt/SG-D3QN) and [SG-D3QN-intrinsic](https://github.com/dmartinezbaselga/intrinsic-rewards-navigation).  The authors are thankful for their works and for making them available.

## Citation
If you use this work in your own research or wish to refer to the paper's results, please use the following BibTeX entries.
```bibtex
@inproceedings{damely24hyp2nav,
  author	= {Di Melendugno, Guido Maria D'Amely and Flaborea, Alessandro and Mettes, Pascal and Galasso, Fabio},
  booktitle	= {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title		= {Hyp2Nav: Hyperbolic Planning and Curiosity for Crowd Navigation}, 
  year		= {2024},
  url           = {https://arxiv.org/abs/2407.13567}
}

