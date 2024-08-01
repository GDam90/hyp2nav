from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs import policy
from crowd_sim.envs.utils.action import ActionVW
import pdb

class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.rotation_constraint = getattr(config, section).rotation_constraint

    def act(self, ob, args=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        state.w = self.w
        if args:
            action, action_index, returns, not_best_actions, max_action_indexes, radius = self.policy.predict(state, args=args)
            if isinstance(action, ActionVW):
                self.w = action.w
            return action, action_index, returns, not_best_actions, max_action_indexes, radius

        else:
            action, action_index = self.policy.predict(state)
            if isinstance(action, ActionVW):
                self.w = action.w
            return action, action_index

    def get_state(self, ob):
        state = JointState(self.get_full_state(), ob)
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        return self.policy.transform(state)
