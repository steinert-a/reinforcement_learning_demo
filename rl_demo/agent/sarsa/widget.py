import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox, QHBoxLayout
from PyQt6.QtCore import Qt

from dataclasses import dataclass
from typing import Optional, List

from .log_dialog import PolicyHeatmapDialog

INIT_ACTION_VALUES = 0 # 99999

@dataclass
class Interaction:
    state_0: np.ndarray
    action: int
    reward: float
    state_1: np.ndarray
    terminated: bool

class TdSarsaWidget(QWidget):
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
        self._slider_alpha.setValue(10)

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(30)

        self._slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self._slider_gamma.setRange(0, 100)
        self._slider_gamma.setValue(90)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow(self._check_round, self._spin_places)
        ctrl_layout.addRow("alpha step size", self._slider_alpha)
        ctrl_layout.addRow("epsilon greedy", self._slider_epsilon)
        ctrl_layout.addRow("gamma discount", self._slider_gamma)
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

        self._action_values = {}
        self._action_count = {}
        self._last_interaction = None

        self._table_data.setRowCount(0)
        self.update_table()

    def next_action(self, observation) -> int:
        key = self.make_state_key(observation)
        if np.random.random() < self.get_epsilon():
            action = np.random.randint(0, self._total_actions)
        else:
            action = np.argmax(self._action_values[key])
        return action

    def make_state_key(self, state: np.ndarray):
        use_round = self._check_round.isChecked()
        target_state = state.copy().flatten()

        if use_round:
            places = self._spin_places.value()
            target_state = np.round(target_state, places)
        
        key = tuple(target_state.tolist())

        if key not in self._action_values:
            self._action_values[key] = [INIT_ACTION_VALUES] * self._total_actions
        if key not in self._action_count:
            self._action_count[key] = [0] * self._total_actions

        return key
    
    def reinforcement_learning_update(self, interaction_0: Interaction, interaction_1: Interaction):
        alpha = self.get_alpha()
        gamma = self.get_gamma()
        
        if interaction_0.terminated: 
            return
        
        key_0 = self.make_state_key(interaction_0.state_0) 
        key_1 = self.make_state_key(interaction_0.state_1)
        assert key_1 == self.make_state_key(interaction_1.state_0)

        action_0 = interaction_0.action
        action_1 = interaction_1.action

        reward =  interaction_0.reward

        q_0 = self._action_values[key_0][action_0]
        if interaction_0.terminated:
            q_1 = 0.0 # expected return from terminal state is always 0
        else:
            q_1 = self._action_values[key_1][action_1]

        # reward + gamma * q_1 == q_0 => defines error
        # new q_0 = q_0 + alpha * error 
        self._action_values[key_0][action_0] = q_0 + alpha * (reward + gamma * q_1 - q_0) 
        self._action_count[key_0][action_0] += 1

    def reinforcement_learning(self, observation_0: np.ndarray, action: int, reward: float, observation_1: np.ndarray, terminated: Optional[bool]):
        interaction = Interaction(observation_0, action, reward, observation_1, terminated == True)

        if self._last_interaction is not None:
            self.reinforcement_learning_update(self._last_interaction, interaction)
        
        if interaction.terminated is not None and interaction.terminated == False:
            self._last_interaction = interaction
        else:
            self._last_interaction = None
        self.update_table()

    def on_log_policy_heat_map(self):
        try:
            x_set = set()
            y_set = set()
            r_set = set()
            for key in self._action_values.keys():
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
                        if key in self._action_values:
                            if key not in temp_action_values:
                                temp_action_values[(x,y)] = self._action_values[key]
                            else:
                                a = np.array(self._action_values[key])
                                b = np.array(temp_action_values[(x,y)])
                                temp_action_values[(x,y)] = a + b
                            heat_map_data[yi][xi] = np.argmax(temp_action_values[(x,y)])

            PolicyHeatmapDialog(heat_map_data, sorted_x, sorted_y, self).exec()
        except Exception as exp:
            print("can't calculate heat map: ", exp)