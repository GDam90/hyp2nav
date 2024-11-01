import argparse
import importlib.util
import logging
import os
import time
from math import atan2, cos, pi, sin, sqrt

import gym
import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped, Twist
from jackal_interface import JackalInterface
from lmpcc_msgs.msg import lmpcc_obstacle, lmpcc_obstacle_array
from nav_msgs.msg import Odometry
from pedestrian_sim_interface import PedestrianSimInterface
from tf.transformations import euler_from_quaternion

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.reward_estimate import Reward_Estimator
from crowd_nav.utils.explorer import Explorer
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.socialforce import SocialForce
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


class dumping_info:
    def __init__(self) -> None:
        self.v = []
        self.w = []
        self.px = []
        self.py = []
        self.theta = []
        self.gx = []
        self.gy = []
        self.obs_x = []
        self.obs_y = []
        self.obs_vx = []
        self.obs_vy = []
    
    def append(self, v, w, robot_state: FullState, obstacles):
        return
        self.v.append(v)
        self.w.append(w)
        self.px.append(robot_state.px)
        self.py.append(robot_state.py)
        self.theta.append(robot_state.theta)
        self.gx.append(robot_state.gx)
        self.gy.append(robot_state.gy)
        self.obs_x.append([o.px for o in obstacles])
        self.obs_y.append([o.py for o in obstacles])
        self.obs_vx.append([o.vx for o in obstacles])
        self.obs_vy.append([o.vy for o in obstacles])

    
    def dump_in_file(self, file_name, i):
        return
        with open(file_name, 'a') as f:
            f.write("\nEpisode " + str(i) + ":")
            f.write("\nv: ")
            for v in self.v:
                f.write("{:.2f}, ".format(v))
            f.write("\nw: ")
            for w in self.w:
                f.write("{:.2f}, ".format(w))
            f.write("\npx: ")
            for px in self.px:
                f.write("{:.2f}, ".format(px))
            f.write("\npy: ")
            for py in self.py:
                f.write("{:.2f}, ".format(py))
            f.write("\ntheta: ")
            for theta in self.theta:
                f.write("{:.2f}, ".format(theta))
            f.write("\ngx: ")
            for gx in self.gx:
                f.write("{:.2f}, ".format(gx))
            f.write("\ngy: ")
            for gy in self.gy:
                f.write("{:.2f}, ".format(gy))
            f.write("\nobs_x: ")
            for timestep in self.obs_x:
                f.write("[")
                for o in timestep:
                    f.write("{:.2f}, ".format(o))
                f.write("], ")
            f.write("\nobs_y: ")
            for timestep in self.obs_y:
                f.write("[")
                for o in timestep:
                    f.write("{:.2f}, ".format(o))
                f.write("], ")
            f.write("\nobs_vx: ")
            for timestep in self.obs_vx:
                f.write("[")
                for o in timestep:
                    f.write("{:.2f}, ".format(o))
                f.write("], ")
            f.write("\nobs_vy: ")
            for timestep in self.obs_vx:
                f.write("[")
                for o in timestep:
                    f.write("{:.2f}, ".format(o))
                f.write("], ")

class baseline_planner:
    def __init__(self, policy_name):
        self.robot_policy = None
        self.cur_state = None
        rospy.init_node('baseline_planner_node', anonymous=True)
        self.robot_policy = policy_factory[policy_name]()
        self.peds_full_state = []
        self.obstacle_radius = 0.2
        self.time_step = 0.25
        self.robot_full_state = FullState(0,0,0,0,0,0,0,0,0)
        self.robot_full_state.v_pref = 1.25
        self.robot_full_state.radius = 0.2
        self.current_goal = False
        self.rotating = False
        self.last_state_time_ = rospy.Time.now()
        # self.jackal_interface = JackalInterface()
        self.jackal_interface = PedestrianSimInterface()
        self.filename = "metrics_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        self.metrics = dumping_info()
        self.episode_i = 0

    def start(self):
        self.sub_goal = rospy.Subscriber("/roadmap/goal", PoseStamped, self.goal_callback)
        # self.sub_state = rospy.Subscriber("/Robot_1/pose", PoseStamped, self.state_callback)
        self.sub_state = rospy.Subscriber("/odometry/filtered", Odometry, self.state_callback)
        self.robot_action_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        print("Ros node started")
        rospy.spin()

    def load_policy_model(self, args):
        if not isinstance(self.robot_policy, SocialForce) and not isinstance(self.robot_policy, ORCA):
            device = torch.device(args.device if torch.cuda.is_available() and args.gpu else "cpu")  #! torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"))

            if args.model_dir is not None:
                if args.config is not None:
                    config_file = args.config
                else:
                    config_file = os.path.join(args.model_dir, 'config.py')
                if args.il:
                    model_weights = os.path.join(args.model_dir, 'il_model.pth')
                elif args.rl:
                    if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                        model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
                    else:
                        print(os.listdir(args.model_dir))
                        model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
                else:
                    model_weights = os.path.join(args.model_dir, 'best_val.pth')

            else:
                config_file = args.config

            spec = importlib.util.spec_from_file_location('config', config_file)
            if spec is None:
                parser.error('Config file not found.')
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)

            # configure policy
            policy_config = config.PolicyConfig(False)
            policy = policy_factory[policy_config.name]()
            reward_estimator = Reward_Estimator()
            env_config = config.EnvConfig(False)
            reward_estimator.configure(env_config)
            policy.reward_estimator = reward_estimator
            if args.planning_depth is not None:
                policy_config.model_predictive_rl.do_action_clip = True
                policy_config.model_predictive_rl.planning_depth = args.planning_depth
            if args.planning_width is not None:
                policy_config.model_predictive_rl.do_action_clip = True
                policy_config.model_predictive_rl.planning_width = args.planning_width
            if args.sparse_search:
                policy_config.model_predictive_rl.sparse_search = True

            policy.configure(policy_config, device)
            if policy.trainable:
                if args.model_dir is None:
                    parser.error('Trainable policy must be specified with a model weights directory')
                policy.load_model(model_weights)

            # for continous action
            action_dim = 2
            max_action = [1., 0.5235988]
            min_action = [0., -0.5235988]
            if policy.name == 'TD3RL':
                policy.set_action(action_dim, max_action, min_action)
            self.robot_policy = policy
            policy.set_v_pref(1.25)
            self.robot_policy.set_time_step(self.time_step)
            if not isinstance(self.robot_policy, ORCA) and not isinstance(self.robot_policy, SocialForce):
                self.robot_policy.set_epsilon(0.01)
            policy.set_phase("test")
            policy.set_device(device)

            # set safety space for ORCA in non-cooperative simulation
            if isinstance(self.robot_policy, ORCA):
                self.robot_policy.safety_space = args.safety_space
        else:
            self.robot_policy.time_step = self.time_step
        print("policy loaded")

    
    def process_obstacles(self):
        self.peds_full_state.clear()
        for o in self.jackal_interface.dynamic_obstacles:
            self.peds_full_state.append(ObservableState(px=o.x, py=o.y,
                                                        vx=o.vx, vy=o.vy,
                                                        radius=0.2))

    def goal_callback(self, goal: PoseStamped):
        # print(self.robot_full_state.gx , self.robot_full_state.gy)
        if not self.current_goal:
            self.rotating = True
            self.current_goal = True
        elif sqrt((self.robot_full_state.gx - goal.pose.position.x)**2 + (self.robot_full_state.gy - goal.pose.position.y)**2):
            self.rotating = True
            self.metrics.dump_in_file(self.filename, self.episode_i)
            self.episode_i = self.episode_i = 1
            self.metrics = dumping_info()
        self.robot_full_state.gx = goal.pose.position.x
        self.robot_full_state.gy = goal.pose.position.y

    # def state_callback(self, robot_state: PoseStamped):
    #     if self.last_state_time_ + rospy.Duration(1. / 20.) < robot_state.header.stamp:
    #         self.last_state_time_ = robot_state.header.stamp
    #         action_cmd = Twist()
    #         action_cmd.linear.x = 0.0
    #         action_cmd.linear.y = 0.0
    #         action_cmd.angular.z = 0.0
    #         if self.jackal_interface.enable_output_:
    #             if self.rotating:
    #                 _, _, theta = euler_from_quaternion([robot_state.pose.orientation.x, robot_state.pose.orientation.y, robot_state.pose.orientation.z, robot_state.pose.orientation.w])
    #                 angle_to_goal = atan2(self.robot_full_state.gy - robot_state.pose.position.y, self.robot_full_state.gx - robot_state.pose.position.x)%(2*pi) - theta%(2*pi)
    #                 if abs(angle_to_goal) < 0.4:
    #                     self.rotating = False
    #                 else:
    #                     action_cmd.angular.z = 1.0
    #             elif self.current_goal:
    #                 self.process_obstacles()
    #                 self.robot_full_state.px = robot_state.pose.position.x
    #                 self.robot_full_state.py = robot_state.pose.position.y
    #                 # print(self.robot_full_state.px, self.robot_full_state.py)
    #                 _, _, theta = euler_from_quaternion([robot_state.pose.orientation.x, robot_state.pose.orientation.y,
    #                                                     robot_state.pose.orientation.z, robot_state.pose.orientation.w])
    #                 self.robot_full_state.theta = theta
    #                 # self.robot_full_state.vx = self.jackal_interface.v * cos(self.robot_full_state.theta)
    #                 # self.robot_full_state.vy = self.jackal_interface.v * sin(self.robot_full_state.theta)
    #                 self.cur_state = JointState(self.robot_full_state, self.peds_full_state)
    #                 action_cmd = Twist()

    #                 dis = np.sqrt((self.robot_full_state.px - self.robot_full_state.gx)**2 + (self.robot_full_state.py - self.robot_full_state.gy)**2)
    #                 if dis < 0.3:
    #                     action_cmd.linear.x = 0.0
    #                     action_cmd.linear.y = 0.0
    #                     action_cmd.angular.z = 0.0
    #                     self.current_goal = None
    #                 else:
    #                     robot_action = self.robot_policy.predict(self.cur_state)
    #                     if isinstance(robot_action, ActionXY):
    #                         action_cmd.angular.z = min(pi/1.5, max(-pi/1.5, (atan2(robot_action.vy, robot_action.vx)%(2*pi) - self.robot_full_state.theta%(2*pi))/self.time_step))
    #                         if abs(action_cmd.angular.z) > pi/2:
    #                             action_cmd.linear.x = min(0.2, sqrt(robot_action.vy**2 + robot_action.vx**2))
    #                         else:    
    #                             action_cmd.linear.x = sqrt(robot_action.vy**2 + robot_action.vx**2)
    #                     else:
    #                         robot_action = robot_action[0]
    #                         action_cmd.angular.z = min(pi/1.5, max(-pi/1.5, robot_action.r/self.time_step))
    #                         if abs(action_cmd.angular.z) > pi/2:
    #                             action_cmd.linear.x = min(0.2, robot_action.r/self.time_step)
    #                         else:    
    #                             action_cmd.linear.x = robot_action.v
    #                     self.metrics.append(self.jackal_interface.v, self.jackal_interface.w, self.robot_full_state, self.peds_full_state)
    #         angle = action_cmd.angular.z * self.time_step + self.robot_full_state.theta
    #         self.robot_full_state.vx = action_cmd.linear.x * cos(angle)
    #         self.robot_full_state.vy = action_cmd.linear.x * sin(angle)
    #         # print(action_cmd)
    #         self.robot_action_pub.publish(action_cmd)

    def state_callback(self, robot_state: Odometry):
        self.current_goal = True
        self.robot_full_state.gx = robot_state.pose.pose.position.x + 6.
        self.robot_full_state.gy = 0.
        if self.last_state_time_ + rospy.Duration(1. / 20.) < robot_state.header.stamp:
            self.last_state_time_ = robot_state.header.stamp
            action_cmd = Twist()
            action_cmd.linear.x = 0.0
            action_cmd.linear.y = 0.0
            action_cmd.angular.z = 0.0
            if self.jackal_interface.enable_output_:
                if self.rotating:
                    _, _, theta = euler_from_quaternion([robot_state.pose.pose.orientation.x, robot_state.pose.pose.orientation.y, robot_state.pose.pose.orientation.z, robot_state.pose.pose.orientation.w])
                    angle_to_goal = atan2(self.robot_full_state.gy - robot_state.pose.pose.position.y, self.robot_full_state.gx - robot_state.pose.pose.position.x)%(2*pi) - theta%(2*pi)
                    if abs(angle_to_goal) < 0.4:
                        self.rotating = False
                    else:
                        action_cmd.angular.z = 1.0
                elif self.current_goal:
                    self.robot_full_state.px = robot_state.pose.pose.position.x
                    self.robot_full_state.py = robot_state.pose.pose.position.y
                    self.process_obstacles()
                    # print(self.robot_full_state.px, self.robot_full_state.py)
                    _, _, theta = euler_from_quaternion([robot_state.pose.pose.orientation.x, robot_state.pose.pose.orientation.y,
                                                        robot_state.pose.pose.orientation.z, robot_state.pose.pose.orientation.w])
                    self.robot_full_state.theta = theta
                    # self.robot_full_state.vx = self.jackal_interface.v * cos(self.robot_full_state.theta)
                    # self.robot_full_state.vy = self.jackal_interface.v * sin(self.robot_full_state.theta)
                    self.cur_state = JointState(self.robot_full_state, self.peds_full_state)
                    action_cmd = Twist()

                    dis = np.sqrt((self.robot_full_state.px - self.robot_full_state.gx)**2 + (self.robot_full_state.py - self.robot_full_state.gy)**2)
                    if dis < 0.3:
                        action_cmd.linear.x = 0.0
                        action_cmd.linear.y = 0.0
                        action_cmd.angular.z = 0.0
                        self.current_goal = None
                    else:
                        robot_action = self.robot_policy.predict(self.cur_state)
                        if isinstance(robot_action, ActionXY):
                            action_cmd.angular.z = min(pi/1.5, max(-pi/1.5, (atan2(robot_action.vy, robot_action.vx)%(2*pi) - self.robot_full_state.theta%(2*pi))/self.time_step))
                            if abs(action_cmd.angular.z) > pi/2:
                                action_cmd.linear.x = min(0.2, sqrt(robot_action.vy**2 + robot_action.vx**2))
                            else:    
                                action_cmd.linear.x = sqrt(robot_action.vy**2 + robot_action.vx**2)
                        else:
                            robot_action = robot_action[0]
                            action_cmd.angular.z = min(pi/1.5, max(-pi/1.5, robot_action.r/self.time_step))
                            if abs(action_cmd.angular.z) > pi/2:
                                action_cmd.linear.x = min(0.2, robot_action.r/self.time_step)
                            else:    
                                action_cmd.linear.x = robot_action.v
                        # self.metrics.append(self.jackal_interface.v, self.jackal_interface.w, self.robot_full_state, self.peds_full_state)
            angle = action_cmd.angular.z * self.time_step + self.robot_full_state.theta
            self.robot_full_state.vx = action_cmd.linear.x * cos(angle)
            self.robot_full_state.vy = action_cmd.linear.x * sin(angle)
            # print(action_cmd)
            self.robot_action_pub.publish(action_cmd)
            self.jackal_interface.update_robot(robot_x=self.robot_full_state.px)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='tree_search_rl')
    parser.add_argument('-m', '--model_dir', type=str, default='data/tsrl5rot/tsrl/1')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    try:
        planner = baseline_planner(sys_args.policy)
        planner.load_policy_model(sys_args)
        planner.start()
    except rospy.ROSException:
        pass
