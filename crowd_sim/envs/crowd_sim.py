import logging
import math
import pdb
import random

import gym
import matplotlib
import matplotlib.lines as mlines
import numpy as np
from gym import spaces
from matplotlib import patches
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.state import JointState, tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.goal_factor = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None
        self.test_changing_size = False

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.random_seed = 0
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        # 动作空间: 速度，朝向
        self.action_space = spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([1, np.pi]),
            dtype=np.float32
        )
        self.phase = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.goal_factor = config.reward.goal_factor
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.discomfort_dist = config.reward.discomfort_dist
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

        if self.robot.kinematics == "holonomic":
            # 动作空间: 速度，朝向
            self.action_space = spaces.Box(
                low=np.array([0, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0, -self.robot.rotation_constraint]),
                high=np.array([1, self.robot.rotation_constraint]),
                dtype=np.float32
            )
        logging.info('rotation constraint: {}'.format(self.robot.rotation_constraint))

    def generate_human(self, human=None, non_stop=False, square=False):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if square is False and non_stop is False:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, -px, -py, 0, 0, 0)
        elif square is False and non_stop is True:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px = human.px
                py = human.py
                gx_noise = (np.random.random() - 0.5) * human.v_pref
                gy_noise = (np.random.random() - 0.5) * human.v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    # if norm((px - agent.px, py - agent.py)) == 0.0:
                    #     continue
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)

        elif square is True and non_stop is False:
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gx = np.random.random() * self.square_width * 0.5 * (- sign)
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        elif square is True and non_stop is True:
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            goal_count = 0
            while True:
                goal_count = goal_count + 1
                px = human.px
                py = human.py
                gx = np.random.random() * self.square_width * 0.5 * (- sign)
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    # if norm((px - agent.px, py - agent.py)) == 0.0:
                    #     break
                    if norm((gx - agent.gx, gy - agent.gy))<min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}

        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        self.random_seed = base_seed[phase] + self.case_counter[phase]
        np.random.seed(self.random_seed)
        if self.case_counter[phase] >= 0:
            # np.random.seed(base_seed[phase] + self.case_counter[phase])
            # random.seed(base_seed[phase] + self.case_counter[phase])
            # random.seed(2100)

            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
                # print('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                human_num = self.human_num
            self.humans = []
            for i in range(human_num):
                if self.current_scenario == 'circle_crossing':
                    if human_num > 5 and i > 4:
                        self.humans.append(self.generate_human(square=True))
                    else:
                        self.humans.append(self.generate_human())
                else:
                    self.humans.append(self.generate_human(square=True))

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'traj'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        weight_goal = self.goal_factor
        weight_safe = self.discomfort_penalty_factor
        weight_terminal = 1.0
        re_collision = self.collision_penalty
        re_arrival = self.success_reward
        # collision detection
        dmin = float('inf')
        collision = False
        safety_penalty = 0.0
        num_discom = 0
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human_actions[i].vx - action.vx
                vy = human_actions[i].vy - action.vy
            else:
                vx = human_actions[i].vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human_actions[i].vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
            if closest_dist < dmin:
                dmin = closest_dist
            if closest_dist < self.discomfort_dist:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist)
                num_discom = num_discom + 1
            # dis_begin = np.sqrt(px**2 + py**2) - human.radius - self.robot.radius
            # dis_end = np.sqrt(ex**2 + ey**2) - human.radius - self.robot.radius
            # penalty_begin = 0
            # penalty_end = 0
            # if dis_begin < self.discomfort_dist:
            #     penalty_begin = dis_begin - self.discomfort_dist
            # if dis_end < self.discomfort_dist:
            #     penalty_end = dis_end - self.discomfort_dist
            # safety_penalty = safety_penalty + (penalty_end - penalty_begin)

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        cur_position = np.array((self.robot.px, self.robot.py))
        goal_position = np.array(self.robot.get_goal_position())
        reward_goal = (norm(cur_position - goal_position) - norm(end_position - goal_position))
        reaching_goal = norm(end_position - goal_position) < self.robot.radius
        delta_w = 0.0
        if delta_w < 0.5:
            reward_omega = -0.01 * (0.5 - delta_w) * (0.5 - delta_w)
        else:
            reward_omega = 0.0
        reward_col = 0.0
        reward_arrival = 0.0
        if self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collision:
            reward_col = re_collision
            done = True
            info = Collision()
        elif reaching_goal:
            reward_arrival = re_arrival
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            done = False
            info = Discomfort(dmin)
            info.num = num_discom
        else:
            done = False
            # info = Nothing()
            info = Discomfort(dmin)
            info.num = num_discom

        reward_terminal = reward_arrival + reward_col
        reward = weight_terminal * reward_terminal + weight_goal * reward_goal + weight_safe * safety_penalty

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    human.reach_count = human.reach_count + 1
                    if human.reach_count == 2:
                        if self.current_scenario == 'circle_crossing':
                            self.generate_human(human, non_stop=True)
                            human.reach_count = 0
                        else:
                            self.generate_human(human, non_stop=True, square=True)
                            human.reach_count = 0

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info

    def peds_predict(self, agent_states, robot_state):
        if self.robot.visible:
            agent_states.append(robot_state)
            human_actions = self.centralized_planner.predict(agent_states)[:-1]
        else:
            human_actions = self.centralized_planner.predict(agent_states)
        return human_actions

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                if self.test_changing_size is False:
                    ob.append(human.get_observable_state())
                else:
                    dis2 = (human.px - agent.px) * (human.px - agent.px) + (human.py - agent.py) * (human.py - agent.py)
                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                        ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode='video', output_file=None, acts=None, not_best_actions=None, all_returns=None, 
               action_indexes=None, max_action_indexes=None, radii=None,feats=None, test_case=None):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.animation import PillowWriter


        self.acts = acts
        self.action_indexes = action_indexes
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        cmap2 = plt.cm.get_cmap('tab10', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=2, tail_width=.6)

            # patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.axis('equal')
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.panel_width/2, self.panel_width/2)
            ax.set_ylim(-self.panel_height/2, self.panel_height/2)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap2(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=10)
                ax.add_artist(human_goal)
                for j in range(len(human.start_pos)):
                    pos = human.start_pos[j]
                    if j ==0:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker=4,
                                                   linestyle='None', markersize=10)
                    else:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker='s',
                                                   linestyle='None', markersize=10)
                    ax.add_artist(test_start)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                #                             color=human_colors[i],
                #                             marker='s', linestyle='None', markersize=10)
                # ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius/4, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius/4, fill=True, color=human_colors[i])
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 5 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0]+0.1, agents[i].center[1]+0.1,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=12) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.arrow(self.states[k - 1][0].px, self.states[k - 1][0].py,
                                              self.states[k][0].px - self.states[k - 1][0].px,
                                              self.states[k][0].py - self.states[k - 1][0].py,
                                            length_includes_head=True, head_width=0.08, lw=0.8, color='black')
                    human_directions = [plt.arrow(self.states[k - 1][1][i].px, self.states[k - 1][1][i].py, self.states[k][1][i].px - self.states[k - 1][1][i].px,self.states[k][1][i].py - self.states[k - 1][1][i].py,
                              length_includes_head=True, head_width=0.08, lw=0.5,
                              color=human_colors[i]) for i in range(self.human_num)]
                    #
                    # human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                    #                                (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                    #                                color=human_colors[i], ls='solid')
                    #                     for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13, 15))
            ax1.tick_params(labelsize=12)
            ax1.set_xlim(-self.panel_width/2, 13)#! self.panel_width/2)   #! /2
            ax1.set_ylim(-self.panel_height/2, self.panel_height/2)
            ax1.set_xlabel('x(m)', fontsize=14)
            ax1.set_ylabel('y(m)', fontsize=14)
        
            # ax2.tight_layout()  # Adjust layout

            show_human_start_goal = False
            robot_color = 'black'
            # add human start positions and goals
            human_colors = [cmap(20) for i in range(len(self.humans))]
            if False:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax1.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax1.add_artist(human_start)
                    
            robot_positions = [state[0].position for state in self.states]
            # save positions in /home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions
            # np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/robot_positions_{test_case}.npy', robot_positions)


            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=15)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            
            
            # add robot possible actions around it
            # Number of sectors
            num_sectors = 16  # 60 degrees / 3.75 degrees
            
            # need to decide the color of the wedge based on the values in not_best_actions
            cmap = matplotlib.cm.get_cmap('YlOrBr')
            
            initial_actions = not_best_actions[0]
            initial_action_indexes = max_action_indexes[0]
            initial_returns = all_returns[0]
            initial_best_action = acts[0]
            act_to_wedge_id = {}
            for i,act_id in enumerate(initial_action_indexes):
                
                act_to_wedge_id[act_id.numpy() // 5] = [initial_actions[i], initial_best_action, initial_returns[i], cmap(initial_returns[i].detach().numpy())]
            
            # Draw sectors
            wedges = []
            pi = 360
            position = 'small_bottom' # ['small_bottom', 'right_large', None] 
            if position == 'small_bottom':
                x = -3.5
                y = -3.5
                radius_wedge = 1
            elif position == 'right_large':
                x = 0
                y = 8.5
                radius_wedge = 4
            
            for i in range(num_sectors):
                if pi == 60:
                    if i in act_to_wedge_id:
                        wedge = patches.Wedge((x,y), radius_wedge, 60 + (i * 3.75), (i + 1) * 3.75 + 60, edgecolor='black', facecolor=act_to_wedge_id[i][-1])
                    else:
                        wedge = patches.Wedge((x,y), radius_wedge, 60 + (i * 3.75), (i + 1) * 3.75 + 60, edgecolor='black', facecolor='white')
                elif pi == 360:                
                    if i in act_to_wedge_id:
                        wedge = patches.Wedge((x,y), radius_wedge, (i * 22.5), (i + 1) * 22.5, edgecolor='black', facecolor=act_to_wedge_id[i][-1])
                    else:
                        wedge = patches.Wedge((x,y), radius_wedge, (i * 22.5), (i + 1) * 22.5, edgecolor='black', facecolor='white')
                
                wedges.append(wedge)
                ax1.add_artist(wedge)
            
            poincare = plt.Circle([8.5,0], 4, fill=False, color="black")
            ax1.add_artist(poincare)
            center = plt.Circle([8.5,0], 0.05, fill=True, color="black")
            ax1.add_artist(center)

            feats_list_x = [zz.reshape(-1)[0] for zz in feats]
            feats_list_y = [zz.reshape(-1)[1] for zz in feats]
            poinc_pos = plt.Circle([4*feats_list_x[0]+8.5,4*feats_list_y[0]], 0.08, fill=True, color="green",alpha=0.1)
            ax1.add_artist(poinc_pos)

            ax1.add_artist(robot_start)
            # add robot and its goal
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color='red', marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color="blue")
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax1.add_artist(robot)
            ax1.add_artist(goal)
            
            thetas = []
            for state in self.states:
                agent_state = state[0]
                thetas.append(agent_state.theta)


            if len(self.humans) == 0:
                if display_numbers:
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                              '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)]
                # add time annotation
                time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=20, transform=ax1.transAxes)
                ax1.add_artist(time)
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.5*radius * np.cos(theta),
                                                                            agent_state.py + 1.5*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax1.add_artist(arrow)
                global_step = 0
            else:
                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                # np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/human_positions_{test_case}.npy', human_positions)

                humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                          for i in range(len(self.humans))]
                ax1.legend([robot, humans[0], goal], ['Robot', 'Human', 'Goal'], fontsize=14)
                # disable showing human numbers
                if display_numbers:
                    human_numbers = [ax1.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i+1),
                                              color='black', fontsize=12) for i in range(len(self.humans))]
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        if self.test_changing_size is True:
                            robot_attention = [ax1.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                        '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                        fontsize=12)]
                            human_attentions = []
                            count = 0
                            for i in range(len(self.humans)):
                                human = humans[i]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    human_attentions = human_attentions + [
                                        ax1.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                 color='black', fontsize=12)]
                                    count = count + 1
                                else:
                                    human_attentions = human_attentions + [
                                        ax1.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 'n',
                                                 color='red', fontsize=12)]
                            attentions = robot_attention + human_attentions
                        else:
                            attentions =[ax1.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                  '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)] + \
                                        [ax1.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset, '{:.2f}'.format(self.attention_weights[0][i+1]),
                                      color='black',fontsize=12) for i in range(len(self.humans))]
                for i, human in enumerate(humans):
                    ax1.add_artist(human)
                    if display_numbers:
                        ax1.add_artist(human_numbers[i])

                # add time annotation
                time = ax1.text(0.32, 1.02, 'Time: {}'.format(0), fontsize=25, transform=ax1.transAxes)
                ax1.add_artist(time)
                # time2 = ax2.text(0.32, 1.02, 'Radius - Time: {}'.format(0), fontsize=25, transform=ax2.transAxes)
                # ax2.add_artist(time2)

                # visualize attention scores
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                #     attention_scores = [plt.text(-5.5, 5, 'robot {}: {:.2f}'.format(0, self.attention_weights[0][0]),
                #                  fontsize=16)] + [plt.text(-5.5, 5 - 0.5 * (i+1), 'Human {}: {:.2f}'.format(i+1, self.attention_weights[0][i+1]),
                #                  fontsize=16) for i in range(len(self.humans))]

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.5*radius * np.cos(theta),
                                                                            agent_state.py + 1.5*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    
                    if i == 0:               
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                # np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/orientations_{test_case}.npy', orientations)

                for arrow in arrows:
                    ax1.add_artist(arrow)
                global_step = 0

            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)
            #
            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax1.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)

            # add human start positions and goals

            robot_trajs = [plt.Circle(robot_positions[0], self.robot.radius/4, fill=True, color="blue")]
            ax1.add_artist(robot_trajs[-1])
            ticks = [-4, -2, 0, 2, 4, None, None, None, None]
            ticks = np.arange(-6, 6, 2)  # Assuming you have 7 ticks
            # ax1.set_xticks(ticks)

            # ticks[-4:] = None  # Set the last three tick positions to None
            tick_labels = [str(tick) if tick is not None else '' for tick in ticks]
            ax1.set_xticklabels(tick_labels)

            
            ax2.set_xlabel('Time step')  # Set x-axis label
            ax2.set_ylabel('Radius')  # Set y-axis label
            ax2.set_ylim(0, 1.0)
            ax2.set_xlim(0,len(radii))
            # ax2.grid(True)  # Show grid




            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                # nonlocal scores
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                if self.human_num >0:
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        if display_numbers:
                            human_numbers[i].set_position((human.center[0], human.center[1]))
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                    # self_attention_scores = [plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
                    #                                   '{:.2f}'.format(self.attention_weights[0][0]), color='black')]

                ax2.plot(radii[:frame_num], color='green')
                # time2.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                if hasattr(self.robot.policy, 'get_attention_weights'):
                    human_attentions = []
                    count = 0
                    for i in range(self.human_num + 1):
                        if i ==0:
                            attentions[i].set_position((robot.center[0]- 0.05, robot.center[1] - x_offset))
                            attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
                        else:
                            if self.test_changing_size is True:
                                human = humans[i-1]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                        human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][count]))
                                    attentions[i].set_color('black')
                                else:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('n')
                                    attentions[i].set_color('red')
                            else:
                                attentions[i].set_position(
                                    (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
    #                    self_attention_dis = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
    #                                               '{:.2f}'.format(self.attention_weights[0][0]), color='black')
    #                    ax1.add_artist(self_attention_dis)

                for arrow in arrows:
                    arrow.remove()
                    
                if frame_num > 0:
                    act_to_wedge_id = {}
                    try:
                        initial_actions = not_best_actions[frame_num+1]
                        initial_action_indexes = max_action_indexes[frame_num+1]
                        initial_returns = all_returns[frame_num+1]
                        
                        for i,act_id in enumerate(initial_action_indexes):
                            
                            act_to_wedge_id[act_id.numpy() // 5] = [initial_actions[i], initial_returns[i], cmap(initial_returns[i].detach().numpy())]

                    except Exception as e: 
                        print(e)

                    for i in range(num_sectors):
                        # change position of pie slice
                        wedges[i].set_theta1(-210 + (i * 22.5) + thetas[frame_num]*180/np.pi)
                        wedges[i].set_theta2(-210 + (i + 1) * 22.5 + thetas[frame_num]*180/np.pi)

                        if i in act_to_wedge_id:
                            wedges[i].set_facecolor(act_to_wedge_id[i][-1])
                        else:
                            wedges[i].set_facecolor('white')
                        try:

                            if self.action_indexes[frame_num+1] // 5 == i:
                                # wedges[i].set_edgecolor('green')
                                wedges[i].set_hatch('/')
                            else:
                                wedges[i].set_edgecolor('gray')
                                wedges[i].set_hatch('')
                                
                        except:
                            pass

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
                
                try:
                    poinc_pos = plt.Circle([4*feats_list_x[frame_num]+8.5,4*feats_list_y[frame_num]], 0.08, fill=True, color="green",alpha=min(0.1+frame_num/len(feats),1))
                    ax1.add_artist(poinc_pos)
                except Exception as e:
                    print(e)
                    pass
                                
                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=robot_arrow_color,
                                                          arrowstyle=arrow_style)]

                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_arrow_color,
                                                               arrowstyle=arrow_style)])
                for arrow in arrows:
                    ax1.add_artist(arrow)
                

                    
                robot_trajs.append(plt.Circle(robot_positions[frame_num], self.robot.radius/4, fill=True, color="blue"))
                ax1.add_artist(robot_trajs[-1])

                # add robot possible actions around it
                # time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax1.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True
            # anim2.running = True

            if output_file is not None:
                # save as video
                # ffmpeg_writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Me'), bitrate=1800)
                # # writer = ffmpeg_writer(fps=10, metadata=(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=ffmpeg_writer)
                writer = PillowWriter(fps=2)
                anim.save(f'{output_file}.gif', writer=writer)
                # anim2.save(output_file writer=writer)


                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError

    def render_old(self, mode='video', output_file=None, acts=None, not_best_actions=None, all_returns=None, 
               action_indexes=None, max_action_indexes=None, radii=None,feats=None, test_case=None):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.animation import PillowWriter

        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        cmap2 = plt.cm.get_cmap('tab10', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=2, tail_width=.6)

            # patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.axis('equal')
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.panel_width/2, self.panel_width/2)
            ax.set_ylim(-self.panel_height/2, self.panel_height/2)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap2(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=10)
                ax.add_artist(human_goal)
                for j in range(len(human.start_pos)):
                    pos = human.start_pos[j]
                    if j ==0:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker=4,
                                                   linestyle='None', markersize=10)
                    else:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker='s',
                                                   linestyle='None', markersize=10)
                    ax.add_artist(test_start)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                #                             color=human_colors[i],
                #                             marker='s', linestyle='None', markersize=10)
                # ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius/4, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius/4, fill=True, color=human_colors[i])
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 5 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0]+0.1, agents[i].center[1]+0.1,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=12) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.arrow(self.states[k - 1][0].px, self.states[k - 1][0].py,
                                              self.states[k][0].px - self.states[k - 1][0].px,
                                              self.states[k][0].py - self.states[k - 1][0].py,
                                            length_includes_head=True, head_width=0.08, lw=0.8, color='black')
                    human_directions = [plt.arrow(self.states[k - 1][1][i].px, self.states[k - 1][1][i].py, self.states[k][1][i].px - self.states[k - 1][1][i].px,self.states[k][1][i].py - self.states[k - 1][1][i].py,
                              length_includes_head=True, head_width=0.08, lw=0.5,
                              color=human_colors[i]) for i in range(self.human_num)]
                    #
                    # human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                    #                                (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                    #                                color=human_colors[i], ls='solid')
                    #                     for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-self.panel_width/2, self.panel_width/2)
            ax.set_ylim(-self.panel_height/2, self.panel_height/2)
            plt.setp(ax.spines.values(), lw=2)

            # ax.set_xlabel('x(m)', fontsize=14)
            # ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            robot_color = 'black'
            # add human start positions and goals
            human_colors = [cmap(20) for i in range(len(self.humans))]
            if False:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=15)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color='red', marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color="blue")
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)

            if len(self.humans) == 0:
                if display_numbers:
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                              '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)]
                # add time annotation
                # time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=20, transform=ax.transAxes)
                # ax.add_artist(time)
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.5*radius * np.cos(theta),
                                                                            agent_state.py + 1.5*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0
            else:
                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                          for i in range(len(self.humans))]
                plt.legend([robot, humans[0], goal], ['Robot', 'Human', 'Goal'], fontsize=14)
                # disable showing human numbers
                if display_numbers:
                    human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i+1),
                                              color='black', fontsize=12) for i in range(len(self.humans))]
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        if self.test_changing_size is True:
                            robot_attention = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                        '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                        fontsize=12)]
                            human_attentions = []
                            count = 0
                            for i in range(len(self.humans)):
                                human = humans[i]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    human_attentions = human_attentions + [
                                        plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                 color='black', fontsize=12)]
                                    count = count + 1
                                else:
                                    human_attentions = human_attentions + [
                                        plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 'n',
                                                 color='red', fontsize=12)]
                            attentions = robot_attention + human_attentions
                        else:
                            attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                  '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)] + \
                                        [plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset, '{:.2f}'.format(self.attention_weights[0][i+1]),
                                      color='black',fontsize=12) for i in range(len(self.humans))]
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    if display_numbers:
                        ax.add_artist(human_numbers[i])

                # add time annotation
                # time = plt.text(0.32, 1.02, 'Time: {}'.format(0), fontsize=25, transform=ax.transAxes)
                # ax.add_artist(time)

                # visualize attention scores
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                #     attention_scores = [plt.text(-5.5, 5, 'robot {}: {:.2f}'.format(0, self.attention_weights[0][0]),
                #                  fontsize=16)] + [plt.text(-5.5, 5 - 0.5 * (i+1), 'Human {}: {:.2f}'.format(i+1, self.attention_weights[0][i+1]),
                #                  fontsize=16) for i in range(len(self.humans))]

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.5*radius * np.cos(theta),
                                                                            agent_state.py + 1.5*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0

            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)
            #
            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)

            # add human start positions and goals

            robot_trajs = [plt.Circle(robot_positions[0], self.robot.radius/8, fill=True, color="blue")]
            ax.add_artist(robot_trajs[-1])

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                # nonlocal scores
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                if self.human_num >0:
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        if display_numbers:
                            human_numbers[i].set_position((human.center[0], human.center[1]))
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                    # self_attention_scores = [plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
                    #                                   '{:.2f}'.format(self.attention_weights[0][0]), color='black')]
                if hasattr(self.robot.policy, 'get_attention_weights'):
                    human_attentions = []
                    count = 0
                    for i in range(self.human_num + 1):
                        if i ==0:
                            attentions[i].set_position((robot.center[0]- 0.05, robot.center[1] - x_offset))
                            attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
                        else:
                            if self.test_changing_size is True:
                                human = humans[i-1]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                        human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][count]))
                                    attentions[i].set_color('black')
                                else:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('n')
                                    attentions[i].set_color('red')
                            else:
                                attentions[i].set_position(
                                    (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
    #                    self_attention_dis = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
    #                                               '{:.2f}'.format(self.attention_weights[0][0]), color='black')
    #                    ax.add_artist(self_attention_dis)

                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=robot_arrow_color,
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_arrow_color,
                                                               arrowstyle=arrow_style)])
                for arrow in arrows:
                    ax.add_artist(arrow)
                
                robot_trajs.append(plt.Circle(robot_positions[frame_num], self.robot.radius/8, fill=True, color="blue"))
                ax.add_artist(robot_trajs[-1])


                # time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                # save as video
                # ffmpeg_writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Me'), bitrate=1800)
                # # writer = ffmpeg_writer(fps=10, metadata=(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=ffmpeg_writer)
                writer = PillowWriter(fps=2)
                anim.save(f'{output_file}{test_case}.gif', writer=writer)
                # anim2.save(output_file writer=writer)


                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError

    def render_correlation(self, mode='video', output_file=None, acts=None, not_best_actions=None, all_returns=None, 
               action_indexes=None, max_action_indexes=None, radii=None,feats=None, test_case=None):
        import emoji
        import matplotlib.pyplot as plt
        from imojify import imojify
        from matplotlib import animation
        from matplotlib.animation import PillowWriter
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        cmap2 = plt.cm.get_cmap('tab10', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=3, tail_width=.6)

            # patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = False
        color_humans = True
        
        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.axis('equal')
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.panel_width/2, self.panel_width/2)
            ax.set_ylim(-self.panel_height/2, self.panel_height/2)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap2(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=10)
                ax.add_artist(human_goal)
                for j in range(len(human.start_pos)):
                    pos = human.start_pos[j]
                    if j ==0:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker=4,
                                                   linestyle='None', markersize=10)
                    else:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker='s',
                                                   linestyle='None', markersize=10)
                    ax.add_artist(test_start)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                #                             color=human_colors[i],
                #                             marker='s', linestyle='None', markersize=10)
                # ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius/4, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius/4, fill=True, color=human_colors[i])
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 5 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0]+0.1, agents[i].center[1]+0.1,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=12) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.arrow(self.states[k - 1][0].px, self.states[k - 1][0].py,
                                              self.states[k][0].px - self.states[k - 1][0].px,
                                              self.states[k][0].py - self.states[k - 1][0].py,
                                            length_includes_head=True, head_width=0.08, lw=0.8, color='black')
                    human_directions = [plt.arrow(self.states[k - 1][1][i].px, self.states[k - 1][1][i].py, self.states[k][1][i].px - self.states[k - 1][1][i].px,self.states[k][1][i].py - self.states[k - 1][1][i].py,
                              length_includes_head=True, head_width=0.08, lw=0.5,
                              color=human_colors[i]) for i in range(self.human_num)]
                    #
                    # human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                    #                                (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                    #                                color=human_colors[i], ls='solid')
                    #                     for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-self.panel_width/2, self.panel_width/2)
            ax.set_ylim(-self.panel_height/2, self.panel_height/2)
            plt.setp(ax.spines.values(), lw=2)

            # ax.set_xlabel('x(m)', fontsize=14)
            # ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            robot_color = 'black'
            # add human start positions and goals
            human_colors = [cmap(20) for i in range(len(self.humans))]
            if False:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=15)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color='red', marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            # robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color="blue")
            
            # robot_emoji = '\U0001F916'
            # emoji_img = OffsetImage(plt.imread(plt.imread(emoji.emoji_unicode[robot_emoji])), zoom=0.05)
            # ab = AnnotationBbox(emoji_img, robot_positions[0], frameon=False)
            # ax.add_artist(ab)
            
            robot_emoji = u'\U0001F916'
            img = plt.imread(imojify.get_img_path(robot_emoji))
            im = OffsetImage(img, zoom=0.06)
            ab = AnnotationBbox(im, robot_positions[0],  frameon=False, pad=0, boxcoords="offset points")
            ax.add_artist(ab)


            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            # ax.add_artist(robot)
            ax.add_artist(goal)

            if len(self.humans) == 0:
                if display_numbers:
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                              '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)]

                # add time annotation
                # time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=20, transform=ax.transAxes)
                # ax.add_artist(time)
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + 1.8*radius * np.cos(agent_state.theta),
                                                               agent_state.py + 1.8*radius * np.sin(agent_state.theta)))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.8*radius * np.cos(theta),
                                                                            agent_state.py + 1.8*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        # arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                        arrows = [patches.FancyArrowPatch((orientation[0],orientation[1]-0.05), color=robot_arrow_color, arrowstyle=arrow_style)]

                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0
            else:
                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                if color_humans:
                    # use colormap YlOrRd to show the attention weights
                    cmap = plt.get_cmap('YlOrRd')   
                    humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, color=cmap(self.attention_weights[0][i+1]/0.2))
                                for i in range(len(self.humans))]
                else:
                    humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                                for i in range(len(self.humans))]
                    plt.legend([robot, humans[0], goal], ['Robot', 'Human', 'Goal'], fontsize=14)
                # disable showing human numbers
                if display_numbers:
                    human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i+1),
                                              color='black', fontsize=12) for i in range(len(self.humans))]
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        if self.test_changing_size is True:
                            robot_attention = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                        '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                        fontsize=12)]
                            human_attentions = []
                            count = 0
                            for i in range(len(self.humans)):
                                human = humans[i]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    human_attentions = human_attentions + [
                                        plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                 color='black', fontsize=12)]
                                    count = count + 1
                                else:
                                    human_attentions = human_attentions + [
                                        plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 'n',
                                                 color='red', fontsize=12)]
                            attentions = robot_attention + human_attentions
                        else:
                            attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                  '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)] + \
                                        [plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset, '{:.2f}'.format(self.attention_weights[0][i+1]),
                                      color='black',fontsize=12) for i in range(len(self.humans))]


                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    if display_numbers:
                        ax.add_artist(human_numbers[i])

                # add time annotation
                # time = plt.text(0.32, 1.02, 'Time: {}'.format(0), fontsize=25, transform=ax.transAxes)
                # ax.add_artist(time)

                # visualize attention scores
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                #     attention_scores = [plt.text(-5.5, 5, 'robot {}: {:.2f}'.format(0, self.attention_weights[0][0]),
                #                  fontsize=16)] + [plt.text(-5.5, 5 - 0.5 * (i+1), 'Human {}: {:.2f}'.format(i+1, self.attention_weights[0][i+1]),
                #                  fontsize=16) for i in range(len(self.humans))]

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py-0.4), (agent_state.px + 2*radius * np.cos(agent_state.theta),
                                                               agent_state.py + 2.2*radius * np.sin(agent_state.theta)-0.1))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.8*radius * np.cos(theta),
                                                                            agent_state.py + 1.8*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                        # arrows = [patches.FancyArrowPatch(orientation[0][0][0],orientation[0][0][1]-0.05,
                        #                                    orientation[0][1][0],orientation[0][0][1],
                        #                                   color=robot_arrow_color, arrowstyle=arrow_style)]

                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0

            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)
            #
            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)

            # add human start positions and goals

            robot_trajs = [plt.Circle(robot_positions[0], self.robot.radius/8, fill=True, color="blue")]
            ax.add_artist(robot_trajs[-1])

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                # nonlocal scores
                global_step = frame_num
                # robot.center = robot_positions[frame_num]
                ab.xy = robot_positions[frame_num]
                
                if self.human_num >0:
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        if display_numbers:
                            human_numbers[i].set_position((human.center[0], human.center[1]))
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                    # self_attention_scores = [plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
                    #                                   '{:.2f}'.format(self.attention_weights[0][0]), color='black')]
                    
                if color_humans:
                    # use colormap YlOrRd to show the attention weights
                    # humans = [humans[human].set_color(cmap(self.attention_weights[frame_num][human+1]))
                    #             for human in range(len(self.humans))]
                    for human in range(len(self.humans)):
                        humans[human].set_color(cmap(self.attention_weights[frame_num][human+1]/0.2))
                if display_numbers:
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        human_attentions = []
                        count = 0
                        for i in range(self.human_num + 1):
                            if i ==0:
                                attentions[i].set_position((robot.center[0]- 0.05, robot.center[1] - x_offset))
                                attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
                            else:
                                if self.test_changing_size is True:
                                    human = humans[i-1]
                                    dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                        attentions[i].set_position(
                                            (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                        attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][count]))
                                        attentions[i].set_color('black')
                                    else:
                                        attentions[i].set_position(
                                            (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                        attentions[i].set_text('n')
                                        attentions[i].set_color('red')
                                else:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
        #                    self_attention_dis = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
        #                                               '{:.2f}'.format(self.attention_weights[0][0]), color='black')
        #                    ax.add_artist(self_attention_dis)

                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=robot_arrow_color,
                                                          arrowstyle=arrow_style)]

                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_arrow_color,
                                                               arrowstyle=arrow_style)])
                for arrow in arrows:
                    ax.add_artist(arrow)
                
                robot_trajs.append(plt.Circle(robot_positions[frame_num], self.robot.radius/8, fill=True, color="blue"))
                ax.add_artist(robot_trajs[-1])


                # time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                # save as video
                # ffmpeg_writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Me'), bitrate=1800)
                # # writer = ffmpeg_writer(fps=10, metadata=(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=ffmpeg_writer)
                writer = PillowWriter(fps=2)
                anim.save(f'{output_file}{test_case}_corr.gif', writer=writer)
                # anim2.save(output_file writer=writer)


                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError


    def render_original_with_corrEmoji(self, mode='video', output_file=None, acts=None, not_best_actions=None, all_returns=None, 
               action_indexes=None, max_action_indexes=None, radii=None,feats=None, test_case=None):
        import emoji
        import matplotlib.pyplot as plt
        from imojify import imojify
        from matplotlib import animation
        from matplotlib.animation import PillowWriter
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage


        self.acts = acts
        self.action_indexes = action_indexes
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        cmap2 = plt.cm.get_cmap('tab10', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=2, tail_width=.6)

            # patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = False
        color_humans = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.axis('equal')
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.panel_width/2, self.panel_width/2)
            ax.set_ylim(-self.panel_height/2, self.panel_height/2)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap2(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=10)
                ax.add_artist(human_goal)
                for j in range(len(human.start_pos)):
                    pos = human.start_pos[j]
                    if j ==0:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker=4,
                                                   linestyle='None', markersize=10)
                    else:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker='s',
                                                   linestyle='None', markersize=10)
                    ax.add_artist(test_start)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                #                             color=human_colors[i],
                #                             marker='s', linestyle='None', markersize=10)
                # ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius/4, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius/4, fill=True, color=human_colors[i])
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 5 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0]+0.1, agents[i].center[1]+0.1,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=12) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.arrow(self.states[k - 1][0].px, self.states[k - 1][0].py,
                                              self.states[k][0].px - self.states[k - 1][0].px,
                                              self.states[k][0].py - self.states[k - 1][0].py,
                                            length_includes_head=True, head_width=0.08, lw=0.8, color='black')
                    human_directions = [plt.arrow(self.states[k - 1][1][i].px, self.states[k - 1][1][i].py, self.states[k][1][i].px - self.states[k - 1][1][i].px,self.states[k][1][i].py - self.states[k - 1][1][i].py,
                              length_includes_head=True, head_width=0.08, lw=0.5,
                              color=human_colors[i]) for i in range(self.human_num)]
                    #
                    # human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                    #                                (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                    #                                color=human_colors[i], ls='solid')
                    #                     for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13, 15))
            ax1.tick_params(labelsize=12)

            ax1.set_xticks([])
            ax1.set_yticks([])

            ax1.set_xlim(-self.panel_width/2, 13)#! self.panel_width/2)   #! /2
            ax1.set_ylim(-self.panel_height/2, self.panel_height/2)
            # ax1.set_xlabel('x(m)', fontsize=14)
            # ax1.set_ylabel('y(m)', fontsize=14)
            plt.setp(ax1.spines.values(), lw=2)
            # ax2.tight_layout()  # Adjust layout

            show_human_start_goal = False
            robot_color = 'black'
            # add human start positions and goals
            human_colors = [cmap(20) for i in range(len(self.humans))]
            if False:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax1.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax1.add_artist(human_start)
                    
            robot_positions = [state[0].position for state in self.states]
            # save positions in /home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions
            # np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/robot_positions_{test_case}.npy', robot_positions)


            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=15)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax1.add_artist(robot_start)
            
            # add robot possible actions around it
            # Number of sectors
            num_sectors = 16  # 60 degrees / 3.75 degrees
            
            # need to decide the color of the wedge based on the values in not_best_actions
            cmap = matplotlib.cm.get_cmap('YlOrBr')
            
            initial_actions = not_best_actions[0]
            initial_action_indexes = max_action_indexes[0]
            initial_returns = all_returns[0]
            initial_best_action = acts[0]
            act_to_wedge_id = {}
            for i,act_id in enumerate(initial_action_indexes):
                
                act_to_wedge_id[act_id.numpy() // 5] = [initial_actions[i], initial_best_action, initial_returns[i], cmap(initial_returns[i].detach().numpy())]
            
            # Draw sectors
            wedges = []
            pi = 360
            position = 'small_bottom' # ['small_bottom', 'right_large', None] 
            if position == 'small_bottom':
                x = -3.5
                y = -3.5
                radius_wedge = 1
            elif position == 'right_large':
                x = 0
                y = 8.5
                radius_wedge = 4
            
            for i in range(num_sectors):
                if pi == 60:
                    if i in act_to_wedge_id:
                        wedge = patches.Wedge((x,y), radius_wedge, 60 + (i * 3.75), (i + 1) * 3.75 + 60, edgecolor='black', facecolor=act_to_wedge_id[i][-1])
                    else:
                        wedge = patches.Wedge((x,y), radius_wedge, 60 + (i * 3.75), (i + 1) * 3.75 + 60, edgecolor='black', facecolor='white')
                elif pi == 360:                
                    if i in act_to_wedge_id:
                        wedge = patches.Wedge((x,y), radius_wedge, (i * 22.5), (i + 1) * 22.5, edgecolor='black', facecolor=act_to_wedge_id[i][-1])
                    else:
                        wedge = patches.Wedge((x,y), radius_wedge, (i * 22.5), (i + 1) * 22.5, edgecolor='black', facecolor='white')
                
                wedges.append(wedge)
                ax1.add_artist(wedge)
            
            poincare = plt.Circle([8.5,0], 4, fill=False, color="black")
            ax1.add_artist(poincare)
            center = plt.Circle([8.5,0], 0.05, fill=True, color="black")
            ax1.add_artist(center)

            feats_list_x = [zz.reshape(-1)[0] for zz in feats]
            feats_list_y = [zz.reshape(-1)[1] for zz in feats]
            poinc_pos = plt.Circle([4*feats_list_x[0]+8.5,4*feats_list_y[0]], 0.08, fill=True, color="green",alpha=0.1)
            ax1.add_artist(poinc_pos)

            ax1.add_artist(robot_start)
            # add robot and its goal
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color='red', marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            # robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color="blue")
            # ax1.add_artist(robot)
                        
            robot_emoji = u'\U0001F916'
            img = plt.imread(imojify.get_img_path(robot_emoji))
            im = OffsetImage(img, zoom=0.06)
            ab = AnnotationBbox(im, robot_positions[0],  frameon=False, pad=0, boxcoords="offset points")
            ax1.add_artist(ab)

            
            ax1.add_artist(goal)
            
            thetas = []
            for state in self.states:
                agent_state = state[0]
                thetas.append(agent_state.theta)


            if len(self.humans) == 0:
                if display_numbers:
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                              '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)]
                # add time annotation
                time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=20, transform=ax1.transAxes)
                ax1.add_artist(time)
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.5*radius * np.cos(theta),
                                                                            agent_state.py + 1.5*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax1.add_artist(arrow)
                global_step = 0
            else:
                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                # np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/human_positions_{test_case}.npy', human_positions)
                if color_humans:
                    # use colormap YlOrRd to show the attention weights
                    cmap = plt.get_cmap('YlOrRd')   
                    humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, color=cmap(self.attention_weights[0][i+1]/0.2))
                                for i in range(len(self.humans))]
                else:
                    humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                            for i in range(len(self.humans))]
                    ax1.legend([robot, humans[0], goal], ['Robot', 'Human', 'Goal'], fontsize=14)
                # disable showing human numbers
                if display_numbers:
                    human_numbers = [ax1.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i+1),
                                              color='black', fontsize=12) for i in range(len(self.humans))]
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        if self.test_changing_size is True:
                            robot_attention = [ax1.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                        '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                        fontsize=12)]
                            human_attentions = []
                            count = 0
                            for i in range(len(self.humans)):
                                human = humans[i]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    human_attentions = human_attentions + [
                                        ax1.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                 color='black', fontsize=12)]
                                    count = count + 1
                                else:
                                    human_attentions = human_attentions + [
                                        ax1.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 'n',
                                                 color='red', fontsize=12)]
                            attentions = robot_attention + human_attentions
                        else:
                            attentions =[ax1.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                  '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)] + \
                                        [ax1.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset, '{:.2f}'.format(self.attention_weights[0][i+1]),
                                      color='black',fontsize=12) for i in range(len(self.humans))]
                for i, human in enumerate(humans):
                    ax1.add_artist(human)
                    if display_numbers:
                        ax1.add_artist(human_numbers[i])

                # add time annotation
                time = ax1.text(0.32, 1.02, 'Time: {}'.format(0), fontsize=25, transform=ax1.transAxes)
                ax1.add_artist(time)
                # time2 = ax2.text(0.32, 1.02, 'Radius - Time: {}'.format(0), fontsize=25, transform=ax2.transAxes)
                # ax2.add_artist(time2)

                # visualize attention scores
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                #     attention_scores = [plt.text(-5.5, 5, 'robot {}: {:.2f}'.format(0, self.attention_weights[0][0]),
                #                  fontsize=16)] + [plt.text(-5.5, 5 - 0.5 * (i+1), 'Human {}: {:.2f}'.format(i+1, self.attention_weights[0][i+1]),
                #                  fontsize=16) for i in range(len(self.humans))]

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics == 'unicycle' and i == 0:
                            direction = (
                            (agent_state.px, agent_state.py-0.4), (agent_state.px + 2*radius * np.cos(agent_state.theta),
                                                               agent_state.py + 2.2*radius * np.sin(agent_state.theta)-0.1))
                        else:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + 1.8*radius * np.cos(theta),
                                                                            agent_state.py + 1.8*radius * np.sin(theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                        # arrows = [patches.FancyArrowPatch(orientation[0][0][0],orientation[0][0][1]-0.05,
                        #                                    orientation[0][1][0],orientation[0][0][1],
                        #                                   color=robot_arrow_color, arrowstyle=arrow_style)]

                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                # np.save(f'/home/aflabor/HypeRL/crowd_nav/data/360_HyperVnet_HHICM_embDim=from32to2_human10/positions/orientations_{test_case}.npy', orientations)

                for arrow in arrows:
                    ax1.add_artist(arrow)
                global_step = 0

            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)
            #
            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax1.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)

            # add human start positions and goals

            robot_trajs = [plt.Circle(robot_positions[0], self.robot.radius/8, fill=True, color="blue")]
            ax1.add_artist(robot_trajs[-1])
            ticks = [-4, -2, 0, 2, 4, None, None, None, None]
            ticks = np.arange(-6, 6, 2)  # Assuming you have 7 ticks
            # ax1.set_xticks(ticks)

            # ticks[-4:] = None  # Set the last three tick positions to None
            # tick_labels = [str(tick) if tick is not None else '' for tick in ticks]
            # ax1.set_xticklabels(tick_labels)

            
            ax2.set_xlabel('Time', fontsize=30)  # Set x-axis label
            ax2.set_ylabel('Radius', fontsize=30)  # Set y-axis label
            ax2.set_ylim(0, 1.0)
            ax2.set_xlim(0,len(radii))
            plt.setp(ax2.spines.values(), lw=2)
            # ax2.grid(True)  # Show grid
            # set fontsize of xticks 

            times = []
            for i in range(0,len(radii),5):
                times.append(str(i * self.time_step + 0.25))
            ax2.set_xticklabels(times)
            ax2.tick_params(axis='both', which='major', labelsize=20)

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                # nonlocal scores
                global_step = frame_num
                # robot.center = robot_positions[frame_num]
                ab.xy = robot_positions[frame_num]
                if self.human_num >0:
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        if display_numbers:
                            human_numbers[i].set_position((human.center[0], human.center[1]))
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                    # self_attention_scores = [plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
                    #                                   '{:.2f}'.format(self.attention_weights[0][0]), color='black')]
                if color_humans:
                    # use colormap YlOrRd to show the attention weights
                    # humans = [humans[human].set_color(cmap(self.attention_weights[frame_num][human+1]))
                    #             for human in range(len(self.humans))]
                    for human in range(len(self.humans)):
                        humans[human].set_color(cmap(self.attention_weights[frame_num][human+1]/0.2))

                ax2.plot(radii[:frame_num], color='green')
                # time2.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                if display_numbers:

                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        human_attentions = []
                        count = 0
                        for i in range(self.human_num + 1):
                            if i ==0:
                                attentions[i].set_position((robot.center[0]- 0.05, robot.center[1] - x_offset))
                                attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
                            else:
                                if self.test_changing_size is True:
                                    human = humans[i-1]
                                    dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                        attentions[i].set_position(
                                            (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                        attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][count]))
                                        attentions[i].set_color('black')
                                    else:
                                        attentions[i].set_position(
                                            (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                        attentions[i].set_text('n')
                                        attentions[i].set_color('red')
                                else:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
        #                    self_attention_dis = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
        #                                               '{:.2f}'.format(self.attention_weights[0][0]), color='black')
        #                    ax1.add_artist(self_attention_dis)

                for arrow in arrows:
                    arrow.remove()
                    
                if frame_num > 0:
                    act_to_wedge_id = {}
                    try:
                        initial_actions = not_best_actions[frame_num+1]
                        initial_action_indexes = max_action_indexes[frame_num+1]
                        initial_returns = all_returns[frame_num+1]
                        
                        for i,act_id in enumerate(initial_action_indexes):
                            
                            act_to_wedge_id[act_id.numpy() // 5] = [initial_actions[i], initial_returns[i], cmap(initial_returns[i].detach().numpy())]

                    except Exception as e: 
                        print(e)

                    for i in range(num_sectors):
                        # change position of pie slice
                        wedges[i].set_theta1(-210 + (i * 22.5) + thetas[frame_num]*180/np.pi)
                        wedges[i].set_theta2(-210 + (i + 1) * 22.5 + thetas[frame_num]*180/np.pi)

                        if i in act_to_wedge_id:
                            wedges[i].set_facecolor(act_to_wedge_id[i][-1])
                        else:
                            wedges[i].set_facecolor('white')
                        try:

                            if self.action_indexes[frame_num+1] // 5 == i:
                                # wedges[i].set_edgecolor('green')
                                wedges[i].set_hatch('/')
                            else:
                                wedges[i].set_edgecolor('gray')
                                wedges[i].set_hatch('')
                                
                        except:
                            pass

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
                
                try:
                    poinc_pos = plt.Circle([4*feats_list_x[frame_num]+8.5,4*feats_list_y[frame_num]], 0.08, fill=True, color="green",alpha=min(0.1+frame_num/len(feats),1))
                    ax1.add_artist(poinc_pos)
                except Exception as e:
                    print(e)
                    pass
                                
                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=robot_arrow_color,
                                                          arrowstyle=arrow_style)]

                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_arrow_color,
                                                               arrowstyle=arrow_style)])
                for arrow in arrows:
                    ax1.add_artist(arrow)
                

                    
                robot_trajs.append(plt.Circle(robot_positions[frame_num], self.robot.radius/4, fill=True, color="blue"))
                ax1.add_artist(robot_trajs[-1])

                # add robot possible actions around it
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax1.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True
            # anim2.running = True
            # ax2.set_xticklabels(self.times)

            if output_file is not None:
                # save as video
                # ffmpeg_writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Me'), bitrate=1800)
                # # writer = ffmpeg_writer(fps=10, metadata=(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=ffmpeg_writer)
                writer = PillowWriter(fps=2)
                anim.save(f'{output_file}{test_case}.gif', writer=writer)
                # anim2.save(output_file writer=writer)


                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError
