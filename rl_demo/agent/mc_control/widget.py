import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox
from PyQt6.QtCore import Qt

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Interaction:
    state_0: np.ndarray
    action: int
    reward: float
    state_1: np.ndarray



class MonteCarloControlWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset([])
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._check_episode_steps = QCheckBox("max episode steps")
        self._check_episode_steps.setChecked(False)

        self._spin_episode_steps = QSpinBox()
        self._spin_episode_steps.setEnabled(False)
        self._spin_episode_steps.setMinimum(1)
        self._spin_episode_steps.setMaximum(999)
        self._spin_episode_steps.setValue(10)

        self._check_round = QCheckBox("state decimal places")
        self._check_round.setChecked(False)

        self._spin_places = QSpinBox()
        self._spin_places.setEnabled(False)
        self._spin_places.setMinimum(0)
        self._spin_places.setMaximum(9)
        self._spin_places.setValue(1)

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(50)

        self._slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self._slider_gamma.setRange(0, 100)
        self._slider_gamma.setValue(50)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow(self._check_round, self._spin_places)
        ctrl_layout.addRow(self._check_episode_steps, self._spin_episode_steps)
        ctrl_layout.addRow("epsilon soft policy", self._slider_epsilon)
        ctrl_layout.addRow("gamma", self._slider_gamma)
        layout_main.addLayout(ctrl_layout)

        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["state", "action", "count", "value"])
        layout_main.addWidget(self._table_data)

        # connections
        self._check_round.toggled.connect(self.on_round_toggled)
        self._check_episode_steps.toggled.connect(self.on_episode_steps_toggled)

    def on_episode_steps_toggled(self, checked: bool):
        self._spin_episode_steps.setEnabled(checked)

    def on_round_toggled(self, checked: bool):
        self._spin_places.setEnabled(checked)

    def update_table(self):
        self._table_data.setRowCount(len(self._action_values.keys()) * len(self._action_space))

        row = 0
        for key in self._action_values.keys():
            for action in range(len(self._action_space)):
                self._table_data.setItem(
                        row,
                        0,
                        QTableWidgetItem(str(key))
                )
                self._table_data.setItem(
                        row,
                        1,
                        QTableWidgetItem(self._action_space[action])
                )
                self._table_data.setItem(
                        row,
                        2,
                        QTableWidgetItem(str(self._action_count[key][action]))
                )
                self._table_data.setItem(
                        row,
                        3,
                        QTableWidgetItem(str(self._action_values[key][action]))
                )
                row += 1


    def get_epsilon(self):
        value = self._slider_epsilon.value()
        return value/100.0

    def get_gamma(self):
        value = self._slider_gamma.value()
        return value/100.0
    
    def reset(self, action_space):
        self._action_space = action_space
        self._total_actions = len(self._action_space)

        self._action_values = {}
        self._action_count = {}
        self._policy = {}
        self._returns = {}
        self._episode:List[Interaction] = []

        self._table_data.setRowCount(0)
        self.update_table()

    def get_epsilon_soft_policy(self, state):
        epsilon = self.get_epsilon()
        action_values = self.get_action_values(state)
        best_action = np.argmax(action_values)

        policy = np.full(self._total_actions, epsilon / self._total_actions)
        policy[best_action] = policy[best_action] + 1.0 - epsilon
        return policy

    def next_action(self, observation) -> int:
        policy = self.get_epsilon_soft_policy(observation)
        return np.random.choice(self._total_actions, p=policy)

    def make_state_key(self, state: np.ndarray):
        use_round = self._check_round.isChecked()
        places = self._spin_places.value()

        target_state = state.copy().flatten()

        if use_round:
            target_state = np.round(target_state, places)

        return tuple(target_state.tolist())
    
    def get_action_values(self, state):
        key = self.make_state_key(state)
        if key not in self._action_values:
            self._action_values[key] = [0] * self._total_actions

        return self._action_values[key]

    def reinforcement_learning_update(self):
        gamma = self.get_gamma()
        ret = 0
        for interaction in reversed(self._episode):
            ret = gamma * ret + interaction.reward
            key_0 = self.make_state_key(interaction.state_0)

            if key_0 not in self._returns:
                self._returns[key_0] = [[]] * self._total_actions
            self._returns[key_0][interaction.action].append(ret)

            if key_0 not in self._action_count:
                self._action_count[key_0] = [0] * self._total_actions
            self._action_count[key_0][interaction.action] += 1  

            if key_0 not in self._action_values:
                self._action_values[key_0] = [0] * self._total_actions
            self._action_values[key_0][interaction.action] = np.mean(self._returns[key_0][interaction.action])
        
        self._episode = []
        self.update_table()       

    def reinforcement_learning(self, observation_0: np.ndarray, action: int, reward: float, observation_1: np.ndarray, terminated: Optional[bool]):
        interaction = Interaction(observation_0, action, reward, observation_1)
        self._episode.append(interaction)

        if terminated or (self._check_episode_steps.isChecked() and self._spin_episode_steps.value() <= len(self._episode)):
            self.reinforcement_learning_update()
