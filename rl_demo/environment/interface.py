import numpy as np
from PyQt6.QtWidgets import QWidget
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional


class IEnvironment(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def widget(self) -> QWidget:
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def action_space(self) -> List[str]:
        """returns action space"""
        pass

    @abstractmethod
    def state(self) -> Tuple[np.ndarray, Optional[bool]]:
        """returns observation, terminated"""
        pass

    @abstractmethod
    def step(self, action:int) -> Tuple[np.ndarray, float, Optional[bool]]:
        """returns observation, reward, terminated"""
        pass