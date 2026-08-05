import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox
from PyQt6.QtCore import Qt

class PolicyIterationWidget(QWidget):
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

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(50)

        self._slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self._slider_gamma.setRange(0, 100)
        self._slider_gamma.setValue(50)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow(self._check_round, self._spin_places)
        ctrl_layout.addRow("epsilon", self._slider_epsilon)
        ctrl_layout.addRow("gamma", self._slider_gamma)
        layout_main.addLayout(ctrl_layout)

        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["state", "action", "count", "value"])
        layout_main.addWidget(self._table_data)

        self._check_round.toggled.connect(self.on_round_toggled)

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
                        QTableWidgetItem(str(tuple(float(x) for x in key)))
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
        self._action_values = {}
        self._action_count = {}
        self._table_data.setRowCount(0)
        self.update_table() 

    def next_action(self, observation) -> int:
        if np.random.random() < self.get_epsilon():
            action = np.random.randint(0, len(self._action_space))
        else:
            action_values = self.get_action_values(observation)
            action = np.argmax(action_values)
        return action

    def make_state_key(self, state: np.ndarray):
        use_round = self._check_round.isChecked()
        places = self._spin_places.value()

        target_state = state.copy().flatten()

        if use_round:
            target_state = np.round(target_state, places)

        return tuple(target_state)
    
    def get_action_values(self, state):
        key = self.make_state_key(state)
        if key not in self._action_values:
            self._action_values[key] = [0] * len(self._action_space)

        return self._action_values[key]          

    def reinforcement_learning(self, observation_0: np.ndarray, action: int, reward: float):
        key = self.make_state_key(observation_0)
        gamma = self.get_gamma()

        if key not in self._action_values:
            self._action_values[key] = [0] * len(self._action_space)

        self._action_values[key][action] = reward + gamma * self._action_values[key][action]

        if key not in self._action_count:
            self._action_count[key] = [0] * len(self._action_space)
        
        self._action_count[key][action] += 1
        
        self.update_table()
