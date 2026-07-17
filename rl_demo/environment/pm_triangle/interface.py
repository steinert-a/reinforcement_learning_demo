import numpy as np
from typing import List, Tuple, Optional

from .widget import PmTriangleWidget, Actions

from ..interface import IEnvironment

class EnvPmTriangle(IEnvironment):

    def __init__(self, args):
        self._args = args
        self._env_widget = PmTriangleWidget()
    
    def name(self):
        return "Project Management Triangle"

    def widget(self):
        return self._env_widget
    
    def reset(self):
        self._env_widget.reset()

    def action_space(self):
        return [a.value for a in Actions]
    
    def state(self):
        return self._env_widget.get_state(), None

    def step(self, action):
        self._env_widget.perform_action(action)
        return self._env_widget.get_state(), self._env_widget.get_reward(), None


