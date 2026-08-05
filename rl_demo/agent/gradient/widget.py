import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QDoubleSpinBox, QCheckBox
from PyQt6.QtCore import Qt


def softmax(x):
    x = np.array(x, dtype=float)
    if x.size == 0:
        return []
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class GradientWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset([])
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._check_alpha = QCheckBox("alpha")
        self._check_alpha.setChecked(False)
        self._slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self._slider_alpha.setRange(0, 100)
        self._slider_alpha.setValue(50)
        self._slider_alpha.setEnabled(False)

        self._slider_step_size = QSlider(Qt.Orientation.Horizontal)
        self._slider_step_size.setRange(0, 100)
        self._slider_step_size.setValue(50)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow(self._check_alpha, self._slider_alpha)
        ctrl_layout.addRow("step size", self._slider_step_size)
        layout_main.addLayout(ctrl_layout)


        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["action", "count", "strategy", "preference"])
        layout_main.addWidget(self._table_data)

        self._check_alpha.toggled.connect(self.on_toggle_alpha)

    def on_toggle_alpha(self, checked):
        self._slider_alpha.setEnabled(checked)

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
                QTableWidgetItem(str(self._strategy[action]))
        )
        self._table_data.setItem(
                action,
                3,
                QTableWidgetItem(str(self._action_preference[action]))
        )

    def update_table(self, action = None):
        if action is None:
            for row in range(self._action_count):
                self.update_table_row(row)
        else:
            self.update_table_row(action)

    def get_alpha(self):
        value = self._slider_alpha.value()
        return value/100.0

    def get_step_size(self):
        value = self._slider_step_size.value()
        return value/100.0
    
    def reset(self, action_space):
        self._action_count = len(action_space)
        self._action_space = action_space
        self._action_baseline = 0.0
        self._action_preference = [0] * self._action_count
        self._action_used_count = [0] * self._action_count
        self._action_used_total = 0
        self._strategy = softmax(self._action_preference)
        self._table_data.setRowCount(self._action_count)
        self.update_table()

    def next_action(self) -> int:
        return np.random.choice(self._action_count, p=self._strategy)

    def reinforcement_learning(self, action: int, reward: float):
        self._action_used_total += 1
        self._action_used_count[action] += 1
    
        alpha = 1.0 / self._action_used_total
        if self._check_alpha.isChecked():
            alpha = self.get_alpha()
 
        step_size = self.get_step_size()
    
        for a in range(self._action_count):
            if a == action:
                self._action_preference[a] = self._action_preference[a] + step_size * (reward - self._action_baseline) * (1.0 - self._strategy[a])
            else:
                self._action_preference[a] = self._action_preference[a] - step_size * (reward - self._action_baseline) * self._strategy[a]
    
        self._strategy = softmax(self._action_preference)
        self._action_baseline = self._action_baseline + alpha * (reward - self._action_baseline)

        self.update_table()
