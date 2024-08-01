import argparse
import copy
import importlib.util
import logging
import os
import re
import shutil
import sys

import git
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.pyplot import plot, savefig

from crowd_nav.policy.curiosity import HHICM, HICM, ICM
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.random_encoder import RE3
from crowd_nav.policy.reward_estimate import Reward_Estimator
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.trainer import MPRLTrainer, TD3RLTrainer, TSRLTrainer, VNRLTrainer
from crowd_sim.envs.utils.robot import Robot
import pdb
import datetime
def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    set_random_seeds(args.randomseed)
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if make_new_dir:
            # create folder with timestamp to the output dir
            add = '_{}'.format(re.sub(r'[^0-9]', '', str(datetime.datetime.now())))
            os.makedirs(args.output_dir + add) 
            
            shutil.copy(args.config, os.path.join(args.output_dir+ add, 'config.py'))
    #     if args.overwrite:
    #         shutil.rmtree(args.output_dir)
    #     else:
    #         shutil.rmtree(args.output_dir)
    else:
        add = ''
        if make_new_dir:
            os.makedirs(args.output_dir)
            shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

    args.config = os.path.join(args.output_dir+add, 'config.py')
    log_file = os.path.join(args.output_dir+add, 'output.log')
    in_weight_file = os.path.join(args.output_dir+add, 'in_model.pth')
    il_weight_file = os.path.join(args.output_dir+add, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir+add, 'rl_model.pth')


    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
    )

    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    logging.info('Current random seed: {}'.format(sys_args.randomseed))
    logging.info('Current safe_weight: {}'.format(sys_args.safe_weight))
    logging.info('Current goal_weight: {}'.format(sys_args.goal_weight))
    logging.info('Current re_collision: {}'.format(sys_args.re_collision))
    logging.info('Current re_arrival: {}'.format(sys_args.re_arrival))
    logging.info('Current config content is :{}'.format(config))
    device = torch.device(args.device if torch.cuda.is_available() and args.gpu else "cpu")  #! torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"))
    logging.info('Using device: %s', device)
    # writer = SummaryWriter(log_dir=args.output_dir)
    writer=None

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    train_config = config.TrainConfig(args.debug)
    policy.embedding_dim = args.embedding_dimension
    policy.hyperbolic = args.hyperbolic
    
    if policy_config.use_noisy_net:
        train_config.train.exploration_alg = "noisy_net"
    policy.set_exploration_alg(train_config.train.exploration_alg)
    policy.configure(policy_config, device, args.embedding_dimension, args.hyperbolic, args.curvature)
    policy.set_device(device)
    
    # configure environment
    env_config = config.EnvConfig(args.debug)
    env_config.reward.collision_penalty = args.re_collision
    env_config.reward.success_reward = args.re_arrival
    env_config.reward.goal_factor = args.goal_weight
    env_config.reward.discomfort_penalty_factor = args.safe_weight
    env_config.sim.human_num = args.human_num

    env = gym.make('CrowdSim-v0') # cosa è env? Classe
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    env.set_robot(robot)

    reward_estimator = Reward_Estimator() # it is a module of policy
    reward_estimator.configure(env_config)
    policy.reward_estimator = reward_estimator
    # for continous action
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    # read training parameters
    # train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_batches = train_config.train.train_batches
    train_episodes = train_config.train.train_episodes
    sample_episodes = train_config.train.sample_episodes #?
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model() # Value estimator of the policy
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer
    print(policy_config.name, train_config.train.exploration_alg)
    if train_config.train.exploration_alg == "curiosity":
        intrinsic_reward_alg = ICM(policy_config, 9, 5, device, 0.0001, train_config.train.scaling_factor, policy_config.name, embedding_dimension=args.embedding_dimension)
    elif train_config.train.exploration_alg == "curiosity_hyperdistance":
        intrinsic_reward_alg = HICM(policy_config, 9, 5, device, 0.0001, train_config.train.scaling_factor, policy_config.name, embedding_dimension=args.embedding_dimension)
    elif train_config.train.exploration_alg == "hypercuriosity":
        intrinsic_reward_alg = HHICM(policy_config, 9, 5, device, 0.0001, train_config.train.scaling_factor, policy_config.name, embedding_dimension=args.embedding_dimension, curvature=args.curvature)
    elif train_config.train.exploration_alg == "random_encoder":
        intrinsic_reward_alg = RE3(policy_config, 9, 5, device, 128, beta=train_config.train.beta, beta_schedule=train_config.train.schedule, 
                                rho=train_config.train.rho, k_nn=train_config.train.knn)
    else:
        intrinsic_reward_alg = None

    if policy_config.name == 'model_predictive_rl':
        trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer, env.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model,
                              intrinsic_reward=intrinsic_reward_alg)
    elif policy_config.name == 'tree_search_rl':
        trainer = TSRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer, env.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model,
                              intrinsic_reward = intrinsic_reward_alg)

    elif policy_config.name == 'gat_predictive_rl':
        trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer, env.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model)
    elif policy_config.name == 'td3_rl':
        policy.set_action(action_dim, max_action, min_action)
        trainer = TD3RLTrainer(policy.actor, policy.critic, policy.state_predictor, memory, device, policy, writer,
                              batch_size, optimizer, env.human_num, reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model)
    else:
        trainer = VNRLTrainer(model, memory, device, policy, batch_size, optimizer, writer, intrinsic_reward=intrinsic_reward_alg)
    explorer = Explorer(env, robot, device, writer, policy_config.use_noisy_net, memory, policy.gamma, target_policy=policy,
                        intrinsic_reward = intrinsic_reward_alg)
    policy.save_model(in_weight_file)


    trainer.update_target_model(model) # In first place, model and target model are the same (tgt_model was None during init)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_rl_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    best_val_reward = -1
    best_val_return = -1
    best_val_model = None

    reward_rec = []
    return_rec = []
    discom_tim_rec = []
    nav_time_rec = []
    total_time_rec = []
    reward_in_last_interval = 0
    return_in_last_interval = 0
    nav_time__in_last_interval = 0
    discom_time_in_last_interval = 0
    total_time_in_last_interval = 0
    eps_count = 0
    fw = open(sys_args.output_dir + '/data.txt', 'w')
    print("%f %f %f %f %f" % (0,0,0,0,0), file=fw)
    while episode < train_episodes: # train_episodes = 10000, start training
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        _, _, nav_time, sum_reward, ave_return, discom_time, total_time = \
            explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        eps_count = eps_count + 1
        reward_in_last_interval = reward_in_last_interval + sum_reward
        return_in_last_interval = return_in_last_interval + ave_return
        nav_time__in_last_interval = nav_time__in_last_interval + nav_time
        discom_time_in_last_interval = discom_time_in_last_interval + discom_time
        total_time_in_last_interval = total_time_in_last_interval + total_time
        interval = 100
        if eps_count % interval == 0:
            reward_rec.append(reward_in_last_interval/100.0)
            return_rec.append(return_in_last_interval/100.0)
            discom_tim_rec.append(discom_time_in_last_interval/100.0)
            nav_time_rec.append(nav_time__in_last_interval/100.0)
            total_time_rec.append(total_time_in_last_interval/100.0)
            logging.info('Train in episode %d reward in last 100 episodes %f %f %f %f %f', eps_count, reward_rec[-1],
                         return_rec[-1], discom_tim_rec[-1], nav_time_rec[-1], total_time_rec[-1])
            print("%f %f %f %f %f" % (reward_rec[-1], return_rec[-1], discom_tim_rec[-1], nav_time_rec[-1],
                                     total_time_rec[-1]), file=fw)
            reward_in_last_interval = 0
            return_in_last_interval = 0
            nav_time__in_last_interval = 0
            discom_time_in_last_interval = 0
            total_time_in_last_interval = 0
            min_reward = (np.min(reward_rec) // 5.0 ) * 5.0
            max_reward = (np.max(reward_rec) // 5.0 + 1) * 5.0
            pos = np.array(range(1, len(reward_rec)+1)) * interval
            plt.plot(pos, reward_rec, color='r', marker='.', linestyle='dashed')
            plt.axis([0, eps_count, min_reward, max_reward])
            savefig(args.output_dir + "/reward_record.jpg")
        explorer.log('train', episode) #! wandb
        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            _, _, _, reward, average_return, _, _ = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and average_return > best_val_return:
                best_val_return = average_return
                best_val_model = copy.deepcopy(policy.get_state_dict())
        # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
                explorer.log('test', episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    fw.close()
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the return: {}'.format(best_val_return))
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='td3_rl')
    parser.add_argument('--config', type=str, default='configs/icra_benchmark/td3.py')
    parser.add_argument('--output_dir', type=str, default='data/output1')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=7)
    parser.add_argument('--human_num', type=int, default=5)
    parser.add_argument('--safe_weight', type=float, default=1.0)
    parser.add_argument('--goal_weight', type=float, default=0.2)
    parser.add_argument('--re_collision', type=float, default=-0.25)
    parser.add_argument('--re_arrival', type=float, default=0.25)

    parser.add_argument('--wandb_entity', type=str, default='pinlab-sapienza')
    parser.add_argument('--wandb_project', type=str, default='hyperona')
    parser.add_argument('--wandb_name', type=str, default='first_trial')
    parser.add_argument('--wandb_group', type=str, default='human_num_10')
    parser.add_argument('--wandb_tags', type=str, default='')
    parser.add_argument('--wandb_notes', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, default='offline')
    parser.add_argument('--hyperbolic', default=False, action='store_true')
    parser.add_argument('--embedding_dimension', type=int, default=128)
    parser.add_argument('--curvature', type=float, default=-1.0)

    # arguments for GCN
    # parser.add_argument('--X_dim', type=int, default=32)
    # parser.add_argument('--layers', type=int, default=2)
    # parser.add_argument('--sim_func', type=str, default='embedded_gaussian')
    # parser.add_argument('--layerwise_graph', default=False, action='store_true')
    # parser.add_argument('--skip_connection', default=True, action='store_true')

    sys_args = parser.parse_args()
    main(sys_args)
