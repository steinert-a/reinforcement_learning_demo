import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt

class SampleAverageWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset([])
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(50)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow("epsilon", self._slider_epsilon)
        layout_main.addLayout(ctrl_layout)

        self._table_data = QTableWidget()
        self._table_data.setColumnCount(3)
        self._table_data.setHorizontalHeaderLabels(["action", "count", "value"])
        layout_main.addWidget(self._table_data)


    def update_table_row(self, action):
        self._table_data.setItem(
                action,
                0,
                QTableWidgetItem(self._action_space[action])
        )
        self._table_data.setItem(
                action,
                1,
                QTableWidgetItem(str(self._action_count[action]))
        )
        self._table_data.setItem(
                action,
                2,
                QTableWidgetItem(str(self._action_values[action]))
        )

    def update_table(self, action = None):
        if action is None:
            for row in range(len(self._action_space)):
                self.update_table_row(row)
        else:
            self.update_table_row(action)

    def get_epsilon(self):
        value = self._slider_epsilon.value()
        return value/100.0
    
    def reset(self, action_space):
        self._action_space = action_space
        self._action_values = [0] * len(action_space)
        self._action_count = [0] * len(action_space)
        self._table_data.setRowCount(len(action_space))
        self.update_table()

        self._slider_epsilon.setValue(50)    

    def next_action(self) -> int:
        if np.random.random() < self.get_epsilon():
            action = np.random.randint(0, len(self._action_space))
        else:
            action = np.argmax(self._action_values)
        return action


    def reinforcement_learning(self, action: int, reward: float):
        if self._action_count[action] == 0:
            self._action_count[action] += 1
            self._action_values[action] = reward
        else:
            self._action_values[action] = self._action_values[action] + 1/self._action_count[action] * (reward - self._action_values[action])
            self._action_count[action] += 1
        
        self.update_table(action=action)
