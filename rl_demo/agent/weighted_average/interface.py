import numpy as np
from typing import List, Tuple, Optional

from .widget import WeightedAverageWidget

from ..interface import IAgent

class AgentWeightedAverage(IAgent):

    def __init__(self, args):
        self._args = args
        self._env_widget = WeightedAverageWidget()
    
    def name(self):
        return "Weighted Average"

    def widget(self):
        return self._env_widget

    def reset(self, observation, action_space):
        self._env_widget.reset(action_space)

    def next_action(self, observation):
        return self._env_widget.next_action()

    def reinforcement_learning(self,observation_0, action, reward, observation_1, terminated):
        self._env_widget.reinforcement_learning(action, reward)