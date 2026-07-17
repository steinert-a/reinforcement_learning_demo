import numpy as np
from typing import List, Tuple, Optional

from .widget import GradientWidget

from ..interface import IAgent

class AgentGradient(IAgent):

    def __init__(self, args):
        self._args = args
        self._env_widget = GradientWidget()
    
    def name(self):
        return "Gradient"

    def widget(self):
        return self._env_widget

    def reset(self, observation, action_space):
        self._env_widget.reset(action_space)

    def next_action(self, observation):
        return self._env_widget.next_action()

    def reinforcement_learning(self,observation_0, action, reward, observation_1, terminated):
        self._env_widget.reinforcement_learning(action, reward)