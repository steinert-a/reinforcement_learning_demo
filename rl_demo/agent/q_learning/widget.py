import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox, QHBoxLayout
from PyQt6.QtCore import Qt

from dataclasses import dataclass
from typing import Optional, List

from analytics.probability import softmax
from .log_dialog import PolicyHeatmapDialog

class QLearningWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset([])
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._check_round = QCheckBox("state decimal places")
        self._check_round.setChecked(False)

        self._spin_places = QSpinBox()
        self._spin_places.setEnabled(False)
        self._spin_places.setMinimum(0)
        self._spin_places.setMaximum(9)
        self._spin_places.setValue(1)

        self._slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self._slider_alpha.setRange(0, 100)
        self._slider_alpha.setValue(30)

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(30)

        self._slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self._slider_gamma.setRange(0, 100)
        self._slider_gamma.setValue(90)

        self._check_double = QCheckBox("double q-learning")
        self._check_double.setChecked(True)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow(self._check_round, self._spin_places)
        ctrl_layout.addRow("alpha step size", self._slider_alpha)
        ctrl_layout.addRow("epsilon greedy", self._slider_epsilon)
        ctrl_layout.addRow("gamma discount", self._slider_gamma)
        ctrl_layout.addRow(self._check_double)
        layout_main.addLayout(ctrl_layout)

        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["state", "action", "count", "value"])
        layout_main.addWidget(self._table_data)

        layout_log = QHBoxLayout()
        self._button_log_policy_heat_map = QPushButton("policy heat map")
        layout_log.addWidget(self._button_log_policy_heat_map)
        layout_main.addLayout(layout_log)

        # connections
        self._check_round.toggled.connect(self.on_round_toggled)
        self._button_log_policy_heat_map.clicked.connect(self.on_log_policy_heat_map)

    def on_round_toggled(self, checked: bool):
        self._spin_places.setEnabled(checked)

    def update_table(self):
        self._table_data.setRowCount(len(self._action_values_0.keys()) * self._total_actions)

        row = 0
        for key in self._action_values_0.keys():
            q_0 = np.asarray(self._action_values_0[key])
            q_1 = np.asarray(self._action_values_1[key])
            q = q_0 + q_1
            
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
                        QTableWidgetItem(str(q[action]))
                )
                row += 1

    def get_alpha(self):
        value = self._slider_alpha.value()
        return value/100.0

    def get_epsilon(self):
        value = self._slider_epsilon.value()
        return value/100.0

    def get_gamma(self):
        value = self._slider_gamma.value()
        return value/100.0
    
    def reset(self, action_space):
        self._action_space = action_space
        self._total_actions = len(self._action_space)

        self._action_values_0 = {}
        self._action_values_1 = {}
        self._action_count = {}
        self._last_interaction = None

        self._table_data.setRowCount(0)
        self.update_table()

    def next_action(self, observation) -> int:
        key = self.make_state_key(observation)
        if np.random.random() < self.get_epsilon():
            action = np.random.randint(0, self._total_actions)
        else:
            q_0 = np.asarray(self._action_values_0[key])
            q_1 = np.asarray(self._action_values_1[key])
            action = np.argmax(q_0 + q_1)
        return action

    def make_state_key(self, state: np.ndarray):
        use_round = self._check_round.isChecked()
        target_state = state.copy().flatten()

        if use_round:
            places = self._spin_places.value()
            target_state = np.round(target_state, places)
        
        key = tuple(target_state.tolist())

        if key not in self._action_values_0:
            self._action_values_0[key] = [0.0] * self._total_actions
        if key not in self._action_values_1:
            self._action_values_1[key] = [0.0] * self._total_actions
        if key not in self._action_count:
            self._action_count[key] = [0] * self._total_actions

        return key

    def reinforcement_learning(self, observation_0: np.ndarray, action: int, reward: float, observation_1: np.ndarray, terminated: Optional[bool]):
        alpha = self.get_alpha()
        gamma = self.get_gamma()
        
        key_0 = self.make_state_key(observation_0) 
        key_1 = self.make_state_key(observation_1)

        if self._check_double.isChecked():
            if np.random.random() < self.get_epsilon():
                q_0 = self._action_values_0[key_0]
                q_1 = self._action_values_1[key_1]
            else:
                q_0 = self._action_values_1[key_0]
                q_1 = self._action_values_0[key_1]
        else:
            q_0 = self._action_values_0[key_0]
            q_1 = self._action_values_0[key_1]
        
        q_0[action] = q_0[action] + alpha * (reward + gamma*q_1[np.argmax(q_1)] - q_0[action])
    
        self._action_count[key_0][action] += 1
        self.update_table()

    def on_log_policy_heat_map(self):
        try:
            x_set = set()
            y_set = set()
            r_set = set()
            for key in self._action_values_0.keys():
                x_set.add(key[0])
                y_set.add(key[1])
                r_set.add(key[2:])

            sorted_x = sorted(x_set)
            sorted_y = sorted(y_set)

            heat_map_data = np.full((len(y_set), len(x_set)), np.nan)
            temp_action_values = {}
            for xi, x in enumerate(sorted_x):
                for yi, y in enumerate(sorted_y):
                    for r in r_set:
                        key = (x,y) + r
                        if key in self._action_values_0:
                            q_0 = np.asarray(self._action_values_0[key])
                            q_1 = np.asarray(self._action_values_1[key])
                            q = q_0 + q_1
                            if key not in temp_action_values:
                                temp_action_values[(x,y)] = q
                            else:
                                a = q
                                b = temp_action_values[(x,y)]
                                temp_action_values[(x,y)] = a + b
                            heat_map_data[yi][xi] = np.argmax(temp_action_values[(x,y)])

            PolicyHeatmapDialog(heat_map_data, sorted_x, sorted_y, self).exec()
        except Exception as exp:
            print("can't calculate heat map: ", exp)