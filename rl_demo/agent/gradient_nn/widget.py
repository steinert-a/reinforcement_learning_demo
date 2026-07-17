import torch
import torch.nn as nn
import torch.nn.functional as tf
import torch.optim as optim

import numpy as np
from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QFormLayout, QTableWidget, QTableWidgetItem, QDoubleSpinBox, QCheckBox
from PyQt6.QtCore import Qt

LEARNING_RATE = 0.01

class Perceptron1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.clear()

    def clear(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        return self.linear1(x)

def nn_loss(x, action, reward, baseline):
    log_probs = torch.log_softmax(x, dim=0)
    log_prob = log_probs[action]
    loss = -(reward - baseline) * log_prob
    return loss

def softmax(x):
    x = np.array(x, dtype=float)
    if x.size == 0:
        return []
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class GradientNnWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset(None, None)
            
    def setup_ui(self):
        layout_main = QVBoxLayout(self)

        self._check_alpha = QCheckBox("alpha")
        self._check_alpha.setChecked(False)
        self._slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self._slider_alpha.setRange(0, 100)
        self._slider_alpha.setValue(50)
        self._slider_alpha.setEnabled(False)

        ctrl_layout = QFormLayout()
        ctrl_layout.addRow(self._check_alpha, self._slider_alpha)
        layout_main.addLayout(ctrl_layout)


        self._table_data = QTableWidget()
        self._table_data.setColumnCount(4)
        self._table_data.setHorizontalHeaderLabels(["action", "count", "policy", "preference"])
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
                QTableWidgetItem(str(self._policy[action]))
        )
        self._table_data.setItem(
                action,
                3,
                QTableWidgetItem(str(self._action_preference[action]))
        )

    def update_table(self, observation, action = None):
        self.update_policy(observation)
        if action is None:
            for row in range(self._action_count):
                self.update_table_row(row)
        else:
            self.update_table_row(action)

    def get_alpha(self):
        value = self._slider_alpha.value()
        return value/100.0

    def reset(self, observation, action_space):
        self._action_count = 0
        self._action_preference = None
        self._action_used_count = None
        self._policy = None
        if action_space is not None:
            self._action_count = len(action_space)
            self._action_used_count = [0] * self._action_count

        self._action_space = action_space
        self._action_baseline = 0.0
        self._action_used_total = 0

        self._nn = None
        self._optimizer= None
        if self._action_count > 0:
            self._nn = Perceptron1(observation.size, self._action_count)
            self._optimizer = optim.Adam(self._nn.parameters(), lr=LEARNING_RATE)

        self._table_data.setRowCount(self._action_count+1)
        self.update_table(observation)

    def update_policy(self, observation):
        if self._nn is not None:
            experience = torch.from_numpy(observation).to(torch.float32).flatten()
            preference_tensor = self._nn(experience)
            self._action_preference = preference_tensor.flatten().detach().numpy()
            self._policy = softmax(self._action_preference)

    def next_action(self) -> int:
        return np.random.choice(self._action_count, p=self._policy)

    def reinforcement_learning(self, observation_0, action: int, reward: float):
        self._action_used_total += 1
        self._action_used_count[action] += 1
    
        alpha = 1.0 / self._action_used_total
        if self._check_alpha.isChecked():
            alpha = self.get_alpha()

        # Training
        experience = torch.from_numpy(observation_0).to(torch.float32).flatten()
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        baseline_tensor = torch.tensor(self._action_baseline, dtype=torch.float32)

        self._optimizer.zero_grad()
        logits_preference = self._nn(experience)
        loss = nn_loss(logits_preference, action, reward_tensor, baseline_tensor)
        loss.backward()
        self._optimizer.step()

        self._action_baseline = self._action_baseline + alpha * (reward - self._action_baseline)
            
        self.update_table(observation_0)
