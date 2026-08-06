import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox, QHBoxLayout
from PyQt6.QtCore import Qt

from dataclasses import dataclass
from typing import Optional, List

from .log_dialog import PolicyHeatmapDialog

class PolicyIterationWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset([])
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(10)

        self._slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self._slider_gamma.setRange(0, 100)
        self._slider_gamma.setValue(90)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow("epsilon", self._slider_epsilon)
        ctrl_layout.addRow("gamma", self._slider_gamma)
        layout_main.addLayout(ctrl_layout)

        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["state", "count", "policy", "values"])
        layout_main.addWidget(self._table_data)

        layout_log = QHBoxLayout()
        self._button_log_policy_heat_map = QPushButton("policy heat map")
        layout_log.addWidget(self._button_log_policy_heat_map)
        layout_main.addLayout(layout_log)

        # connections
        self._button_log_policy_heat_map.clicked.connect(self.on_log_policy_heat_map)

    def update_table(self):
        self._table_data.setRowCount(len(self._state_values.keys()))

        row = 0
        for key in self._state_values.keys():
            self._table_data.setItem(
                    row,
                    0,
                    QTableWidgetItem(str(tuple(float(x) for x in key)))
            )
            self._table_data.setItem(
                    row,
                    1,
                    QTableWidgetItem(str(self._state_count[key]))
            )
            self._table_data.setItem(
                    row,
                    2,
                    QTableWidgetItem(self._action_space[self.get_optimal_key_action(key)])
            )
            self._table_data.setItem(
                    row,
                    3,
                    QTableWidgetItem(str(self._state_values[key]))
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
        self._init_state_value = 0 # 99999
        self._state_values = {}
        self._state_count = {}
        self._state_transition = {}
        self._state_reward = {}
        self._table_data.setRowCount(0)
        self.update_table()
    
    def get_optimal_key_policy(self, key_0):
        gamma = self.get_gamma()
        policy = [float("-inf")] * self._total_actions
        for action in range(self._total_actions):
            key_1 = self._state_transition[key_0][action]
            if key_1 is not None:
                policy[action] = self._state_reward[key_0][action] + gamma * self._state_values[key_1]
        return policy

    def get_optimal_policy(self, state):
        key = self.make_state_key(state)
        return self.get_optimal_key_policy(key)

    def get_optimal_action(self, state):
        policy = self.get_optimal_policy(state)
        action = np.argmax(policy)
        return action

    def get_optimal_key_action(self, key):
        policy = self.get_optimal_key_policy(key)
        action = np.argmax(policy)
        return action
    
    def next_action(self, observation) -> int:
        if np.random.random() < self.get_epsilon():
            action = np.random.randint(0, self._total_actions)
        else:
            action = self.get_optimal_action(observation)
        return action

    def make_state_key(self, state: np.ndarray):
        target_state = state.copy().flatten()
        key = tuple(target_state)

        if key not in self._state_values:
            self._state_values[key] = self._init_state_value
        if key not in self._state_count:
            self._state_count[key] = 0
        if key not in self._state_transition:
            self._state_transition[key] = [None] * self._total_actions
        if key not in self._state_reward:
            self._state_reward[key] = [None] * self._total_actions
            
        return key

    def count_state(self, state):
        key = self.make_state_key(state)
        self._action_count[key] += 1

    def reinforcement_learning(self, observation_0: np.ndarray, action: int, reward: float, observation_1: np.ndarray):
        key_0 = self.make_state_key(observation_0)
        key_1 = self.make_state_key(observation_1)
        gamma = self.get_gamma()

        self._state_values[key_0] = reward + gamma * self._state_values[key_1]
        self._state_count[key_0] += 1
        self._state_transition[key_0][action] = key_1
        self._state_reward[key_0][action] = reward

        self.update_table()


    def on_log_policy_heat_map(self):
        try:
            x_set = set()
            y_set = set()
            r_set = set()
            for key in self._state_values.keys():
                x_set.add(key[0])
                y_set.add(key[1])
                r_set.add(key[2:])

            sorted_x = sorted(x_set)
            sorted_y = sorted(y_set)

            heat_map_data = np.full((len(y_set), len(x_set)), np.nan)

            for xi, x in enumerate(sorted_x):
                for yi, y in enumerate(sorted_y):
                    for r in r_set:
                        key = (x,y) + r
                        if key in self._state_values:
                            heat_map_data[yi][xi] = self.get_optimal_key_action(key)

            PolicyHeatmapDialog(heat_map_data, sorted_x, sorted_y, self).exec()
        except Exception as exp:
            print("can't calculate heat map: ", exp)