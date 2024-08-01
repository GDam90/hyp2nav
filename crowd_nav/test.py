import argparse
import importlib.util
import logging
import math
import os
import pdb

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.animation import PillowWriter

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.reward_estimate import Reward_Estimator
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.utils import select_last_checkpoint
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.robot import Robot


def main(args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S", filename="log.txt")
    device = torch.device(args.device if torch.cuda.is_available() and args.gpu else "cpu")  #! torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"))
    device = "cpu"
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            # model_weights = os.path.join(args.model_dir, 'best_val.pth')
            if args.model == -1:
                model_number = 'rl_model_'+ select_last_checkpoint(args.model_dir)
                print(f'evaluating last model: {model_number}')
            elif args.model > -1:
                model_number = 'rl_model_'+str(args.model)
                print(f'evaluating model: {model_number}')
            else:
                model_number = 'best_val'
            model_weights = os.path.join(args.model_dir, f'{model_number}.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    train_config = config.TrainConfig().train
    if policy_config.use_noisy_net:
        train_config.exploration_alg = "noisy_net"
    policy.set_exploration_alg(train_config.exploration_alg)
    reward_estimator = Reward_Estimator()
    env_config = config.EnvConfig(args.debug)
    reward_estimator.configure(env_config)
    policy.reward_estimator = reward_estimator
    if policy_config.name == "model_predictive_rl":
        if args.planning_depth is not None:
            policy_config.model_predictive_rl.do_action_clip = True
            policy_config.model_predictive_rl.planning_depth = args.planning_depth
        if args.planning_width is not None:
            policy_config.model_predictive_rl.do_action_clip = True
            policy_config.model_predictive_rl.planning_width = args.planning_width
        if args.sparse_search:
            policy_config.model_predictive_rl.sparse_search = True

    policy.configure(policy_config, device, args.embedding_dimension, args.hyperbolic)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)
    pdb.set_trace()
    # configure environment
    env_config = config.EnvConfig(args.debug)

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    if args.square:
        env.test_scenario = 'square_crossing'
    if args.circle:
        env.test_scenario = 'circle_crossing'
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario
    
    print('test scenario: ', env.test_scenario)

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    # for continous action
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    if policy.name == 'TD3RL':
        policy.set_action(action_dim, max_action, min_action)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    explorer = Explorer(env, robot, device, None, policy_config.use_noisy_net, gamma=0.9)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)

    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize:
        if robot.policy.name in ['tree_search_rl']:
            policy.model[2].eval()
        rewards = []
        actions = []
        not_best_actions = []
        all_returns = []
        action_indexes = []
        max_action_indexes = []
        radii = []
        feats = []
        discomfort = []
        entropies = []
        ob = env.reset(args.phase, args.test_case)
        save_image_path = args.video_file
        plot_name = f'radius_trend_{args.test_case}'
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action, action_index, returns, not_best_act, max_action_idxs, radius = robot.act(ob, args=args)
            radii.append(radius[0][0])
            feats.append(radius[0][1])
            # print(action)
            ob, _, done, info = env.step(action)
            
            # print(ob)
            # exit()
            if isinstance(info, Discomfort):
                discomfort.append(info.min_dist)

            if isinstance(info, Timeout):
                _ = _ - 0.25
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            actions.append(action)
            action_indexes.append(action_index)
            not_best_actions.append(not_best_act)
            all_returns.append(returns)
            max_action_indexes.append(max_action_idxs)
            
            # entropy
            p =  torch.softmax(returns, dim=0)
            entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(dim=0).unsqueeze(dim=0) / math.log(10)
            entropies.append(entropy.detach().numpy().reshape(-1))
        # env.step(action)
        # env.step(action)
        # env.step(action)
        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
             * reward for t, reward in enumerate(rewards)])
        radii_list = [zz for zz in radii]
        feats_list_x = [zz.reshape(-1)[0] for zz in feats]
        feats_list_y = [zz.reshape(-1)[1] for zz in feats]
        

        if args.save_positions:
            # saving robot and human positions
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/attention_weights.npy', env.attention_weights)
            robot_positions = [state[0].position for state in env.states]
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/robot_positions.npy', robot_positions)
            human_positions = [[state[1][j].position for j in range(len(env.humans))] for state in env.states]
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/human_positions.npy', human_positions)
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/discomfort.npy', discomfort)
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/entropy.npy', entropies, allow_pickle=True)

            radius = env.robot.radius
            # saving robot and human orientations
            orientations = []
            for i in range(env.human_num + 1):
                orientation = []
                for state in env.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if env.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                        (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                            agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.5*radius * np.cos(theta),
                                                                        agent_state.py + 1.5*radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/orientations.npy', orientations)

            # saving radius
            np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/{args.test_case}/radii.npy', radii)
            
            return 

        # # create a plot of the radius trend on a new figure
        # plt.figure()
        # # increase fig size
        # plt.figure(figsize=(15, 8))
        # max_radius = max(radii_list)
        # min_radius = min(radii_list)
        # plt.box(False)
        # # plt.title('Radius trend')
        # # plt.xlabel('Time step')
        # # plt.ylabel('Radius')
        # plt.xticks([])
        # plt.yticks([])

        # plt.ylim(0, 1.0) #(min_radius - 0.0000000001, max_radius + 0.0000000001)
        # plt.plot(radii_list[:-12])
        # plt.savefig(f'{args.video_file}_{args.test_case}_{plot_name}.png')
        # plt.close()
        # plt.show()

        # plt.figure()
        # plt.title('Poincare Embedding trend')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.xlim(-1.0, 1.0)
        # plt.ylim(-1.0, 1.0) #(min_radius - 0.0000000001, max_radius + 0.0000000001)
        # plt.plot(feats_list_x, feats_list_y)
        # plt.savefig(f'{save_image_path}_{plot_name}_poincare.png')
        # plt.close()
        # plt.show()


        # # Create a function to update the plot for each frame
        # def update(frame):
        #     plt.cla()  # Clear the current plot
        #     plt.plot(feats_list_x[:frame], feats_list_y[:frame], 'o', color='blue', alpha=(len(feats_list_x)-frame)/len(feats_list_x))  # Plot up to the current frame
        #     plt.title('Position in Poincarè trend - Time: {}'.format(frame * 0.25))  # Set title
        #     plt.xlabel('Time step')  # Set x-axis label
        #     plt.ylabel('Radius')  # Set y-axis label
        #     plt.xlim(-1.0, 1.0)
        #     plt.ylim(-1.0, 1.0) #(min_radius - 0.0000000001, max_radius + 0.0000000001)
        #     plt.grid(True)  # Show grid
        #     plt.tight_layout()  # Adjust layout

        # # Create the animation
        # fig = plt.figure(figsize=(8, 4))
        # ani = animation.FuncAnimation(fig, update, frames=len(feats_list_x), repeat=False)

        # # Save the animation as a GIF
        # writer = PillowWriter(fps=2)
        # ani.save(f'{save_image_path}_{plot_name}_poincare.gif', writer=writer)

        # # Show the plot (optional)
        # plt.show()
        
        
        # # Create a function to update the plot for each frame
        # def update(frame):
        #     plt.cla()  # Clear the current plot
        #     plt.plot(radii_list[:frame], color='blue')  # Plot up to the current frame
        #     plt.title('Radius trend - Time: {}'.format(frame * 0.25))  # Set title
        #     plt.xlabel('Time step')  # Set x-axis label
        #     plt.ylabel('Radius')  # Set y-axis label
        #     plt.ylim(0, 1.0)
        #     plt.xlim(0,len(radii_list))
        #     plt.grid(True)  # Show grid
        #     plt.tight_layout()  # Adjust layout

        # # Create the animation
        # fig = plt.figure(figsize=(8, 4))
        # ani = animation.FuncAnimation(fig, update, frames=len(radii_list), repeat=False)

        # # Save the animation as a GIF
        # writer = PillowWriter(fps=2)
        # ani.save(f'{save_image_path}_{plot_name}.gif', writer=writer)

        # # Show the plot (optional)
        # plt.show()       
        
        
        
        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_dir is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, args.video_file)
                args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            # env.render('video', args.video_file)
            # if radii[16] < 0.5:
            #     print(f'saving video {args.test_case}')
            env.render_correlation('video', args.video_file, acts=actions, not_best_actions=not_best_actions, all_returns=all_returns,
                    action_indexes=action_indexes, max_action_indexes=max_action_indexes, radii=radii, feats=feats, test_case = args.test_case)
            env.render_original_with_corrEmoji('video', args.video_file, acts=actions, not_best_actions=not_best_actions, all_returns=all_returns,
                    action_indexes=action_indexes, max_action_indexes=max_action_indexes, radii=radii, feats=feats, test_case = args.test_case)

            # else:
            #     print(f'raddi at timestep 16 = {radii[16]}')
            #     return
            
        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
        positions = []
        velocity_rec = []
        rotation_rec = []
        for i in range(len(actions)):
            positions.append(i)
            action = actions[i]
            velocity_rec.append(action.v)
            rotation_rec.append(action.r)
        plt.plot(positions, velocity_rec, color='r', marker='.', linestyle='dashed')
        plt.plot(positions, rotation_rec, color='b', marker='.', linestyle='dashed')
        plt.show()
    else:
        success_rate, collision_rate, avg_time, tot_rew, avg_return, discom_n, tot_time = explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, args=args)
        print('success rate: ', success_rate)
        print('collision rate: ', collision_rate)
        print('average time: ', avg_time)
        print('total reward: ', tot_rew)
        print('average return: ', avg_return)
        print('discomfort: ', discom_n)
        print('total time: ', tot_time)
        # print(explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True))
        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            plt.close()

    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='tree_search_rl')
    parser.add_argument('-m', '--model_dir', type=str, default='data/tsrl10rot/1')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=10)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=False, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    parser.add_argument('--model', type=int, default=-2)
    parser.add_argument('--hyperbolic', default=False, action='store_true')
    parser.add_argument('--embedding_dimension', type=int, default=128)
    parser.add_argument('--save_positions', default=False, action='store_true')

    sys_args = parser.parse_args()
    main(sys_args)

    sys_args = parser.parse_args()
    main(sys_args)
