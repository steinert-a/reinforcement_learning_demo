import numpy as np
from typing import List, Tuple, Optional

from .widget import RobotRentalWidget, Actions

from ..interface import IEnvironment

class EnvRobotRental(IEnvironment):

    def __init__(self, args):
        self._args = args
        self._env_widget = RobotRentalWidget()
    
    def name(self):
        return "Robot Rental"

    def widget(self):
        return self._env_widget
    
    def reset(self):
        self._env_widget.reset()

    def action_space(self):
        return [a.value for a in Actions]
    
    def state(self):
        return self._env_widget.get_state(), None

    def step(self, action):
        reward = self._env_widget.perform_action(action)
        state = self._env_widget.get_state()
        return state, reward, None


