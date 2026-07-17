import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QDoubleSpinBox
from PyQt6.QtCore import Qt

class WeightedAverageWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset([])
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._slider_epsilon = QSlider(Qt.Orientation.Horizontal)
        self._slider_epsilon.setRange(0, 100)
        self._slider_epsilon.setValue(50)

        self._slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self._slider_alpha.setRange(0, 100)
        self._slider_alpha.setValue(50)

        # upper confidence bound
        self._spin_ucb = QDoubleSpinBox()
        self._spin_ucb.setDecimals(2)
        self._spin_ucb.setSingleStep(0.01)
        self._spin_ucb.setRange(0.0, 100.0)
        self._spin_ucb.setValue(0.0)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow("epsilon", self._slider_epsilon)
        ctrl_layout.addRow("alpha", self._slider_alpha)
        ctrl_layout.addRow("ucb", self._spin_ucb)
        layout_main.addLayout(ctrl_layout)


        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["action", "count", "value", "ucb"])
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
                QTableWidgetItem(str(self._action_used_count[action]))
        )
        self._table_data.setItem(
                action,
                2,
                QTableWidgetItem(str(self._action_values[action]))
        )
        self._table_data.setItem(
                action,
                3,
                QTableWidgetItem(str(self._action_ucb[action]))
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

    def get_alpha(self):
        value = self._slider_alpha.value()
        return value/100.0
    
    def reset(self, action_space):
        self._action_count = len(action_space)
        self._action_space = action_space
        self._action_values = [0] * self._action_count
        self._action_used_count = [0] * self._action_count
        self._action_ucb = [0] * self._action_count
        self._action_total = 0
        self._table_data.setRowCount(self._action_count+1)
        self.update_table()

    def next_action(self) -> int:
        if np.random.random() < self.get_epsilon():
            action = np.random.randint(0, len(self._action_space))
        else:
            action_values_ucb = [value + ucb for value, ucb in zip(self._action_values, self._action_ucb)]
            action = np.argmax(action_values_ucb)

        return action


    def reinforcement_learning(self, action: int, reward: float):
        alpha= self.get_alpha()
        self._action_values[action] = self._action_values[action] + alpha * (reward - self._action_values[action])
        
        self._action_total += 1
        self._action_used_count[action] += 1

        ucb_constant = self._spin_ucb.value()
        if ucb_constant > 0:
            for a in range(self._action_count):
                if self._action_used_count[a] > 0:
                    self._action_ucb[a] = ucb_constant* np.sqrt(np.log(self._action_total) / self._action_used_count[a])
            max_ucb = np.max(self._action_ucb)
            for a in range(self._action_count):
                if self._action_used_count[a] == 0:
                 self._action_ucb[a] = max_ucb
            
            self.update_table()
        else:
            self.update_table(action)
