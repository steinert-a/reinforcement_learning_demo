import numpy as np
from PyQt6.QtWidgets import QWidget
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional


class IAgent(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def widget(self) -> QWidget:
        pass

    @abstractmethod
    def reset(self, observation: np.ndarray, action_space: List[str]):
        pass

    @abstractmethod
    def next_action(self, observation) -> int:
        pass

    @abstractmethod
    def reinforcement_learning(self,observation_0: np.ndarray, action: int, reward: float, observation_1: np.ndarray, terminated: Optional[bool]):
        pass