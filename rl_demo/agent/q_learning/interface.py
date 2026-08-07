import numpy as np
from typing import List, Tuple, Optional

from .widget import QLearningWidget

from ..interface import IAgent

class AgentQLearning(IAgent):

    def __init__(self, args):
        self._args = args
        self._env_widget = QLearningWidget()
    
    def name(self):
        return "Q-Learning"

    def widget(self):
        return self._env_widget

    def reset(self, observation, action_space):
        self._env_widget.reset(action_space)

    def next_action(self, observation):
        return self._env_widget.next_action(observation)

    def reinforcement_learning(self,observation_0, action, reward, observation_1, terminated):
        self._env_widget.reinforcement_learning(observation_0, action, reward, observation_1, terminated)