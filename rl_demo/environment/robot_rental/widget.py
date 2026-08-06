from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QGroupBox, QFormLayout, QCheckBox, QLineEdit, QSpinBox
from PyQt6.QtCore import Qt

import math
import numpy as np
from enum import StrEnum

from .plotter import MplCanvas

ACTION_STEP_SIZE = 1

class Actions(StrEnum):
    ACTION_MOVE_ROBOTS_BA_3 = "move_robots_b_to_a_3"
    ACTION_MOVE_ROBOTS_BA_2 = "move_robots_b_to_a_2"
    ACTION_MOVE_ROBOTS_BA_1 = "move_robots_b_to_a_1"

    ACTION_NONE = "none"    

    ACTION_MOVE_ROBOTS_AB_1 = "move_robots_a_to_b_1"
    ACTION_MOVE_ROBOTS_AB_2 = "move_robots_a_to_b_2"
    ACTION_MOVE_ROBOTS_AB_3 = "move_robots_a_to_b_3"

ACTION_MOVE_ROBOTS_MAX = 3
TOTAL_ROBOTS = 20
MAX_DAILY_ROBOT_REQUESTS = 9

REWARD_SUCCESSFUL_RENT = 10
REWARD_MOVE_ROBOT = -2

class RobotRentalWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.reset()
        self.plot()
    
    def setup_ui(self):
        layout_main = QVBoxLayout(self)
        layout_simulation= QHBoxLayout()

        self._canvas = MplCanvas()

         # simulation
        self._spin_requests_a = QSpinBox()
        self._spin_requests_a.setMinimum(1)
        self._spin_requests_a.setMaximum(MAX_DAILY_ROBOT_REQUESTS)
        self._spin_requests_a.setValue(3)

        self._spin_returns_a = QSpinBox()
        self._spin_returns_a.setMinimum(1)
        self._spin_returns_a.setMaximum(MAX_DAILY_ROBOT_REQUESTS)
        self._spin_returns_a.setValue(3)

        self._edit_robots_a_waiting = QLineEdit("0")
        self._edit_robots_a_waiting.setReadOnly(True)
        self._edit_robots_a_active = QLineEdit("0")
        self._edit_robots_a_active.setReadOnly(True)
        self._edit_robots_a_total = QLineEdit("0")
        self._edit_robots_a_total.setReadOnly(True)

        group_shop_floor_a = QGroupBox("simulation - shop floor a")
        group_shop_floor_a_layout = QFormLayout(group_shop_floor_a)
        group_shop_floor_a_layout.addRow(QLabel("Poisson distribution"))
        group_shop_floor_a_layout.addRow("daily requests", self._spin_requests_a)
        group_shop_floor_a_layout.addRow("daily returns", self._spin_returns_a)
        group_shop_floor_a_layout.addRow("robots waiting", self._edit_robots_a_waiting)
        group_shop_floor_a_layout.addRow("robots active", self._edit_robots_a_active)
        group_shop_floor_a_layout.addRow("robots total", self._edit_robots_a_total)
        layout_simulation.addWidget(group_shop_floor_a)

        self._spin_requests_b = QSpinBox()
        self._spin_requests_b.setMinimum(1)
        self._spin_requests_b.setMaximum(MAX_DAILY_ROBOT_REQUESTS)
        self._spin_requests_b.setValue(4)

        self._spin_returns_b = QSpinBox()
        self._spin_returns_b.setMinimum(1)
        self._spin_returns_b.setMaximum(MAX_DAILY_ROBOT_REQUESTS)
        self._spin_returns_b.setValue(2)

        self._edit_robots_b_waiting = QLineEdit("0")
        self._edit_robots_b_waiting.setReadOnly(True)
        self._edit_robots_b_active = QLineEdit("0")
        self._edit_robots_b_active.setReadOnly(True)
        self._edit_robots_b_total = QLineEdit("0")
        self._edit_robots_b_total.setReadOnly(True)

        group_shop_floor_b = QGroupBox("simulation - shop floor b")
        group_shop_floor_b_layout = QFormLayout(group_shop_floor_b)
        group_shop_floor_b_layout.addRow(QLabel("Poisson distribution"))
        group_shop_floor_b_layout.addRow("daily requests", self._spin_requests_b)
        group_shop_floor_b_layout.addRow("daily returns", self._spin_returns_b)
        group_shop_floor_b_layout.addRow("robots waiting", self._edit_robots_b_waiting)
        group_shop_floor_b_layout.addRow("robots active", self._edit_robots_b_active)
        group_shop_floor_b_layout.addRow("robots total", self._edit_robots_b_total)
        layout_simulation.addWidget(group_shop_floor_b)

        # control
        self._spin_ctrl_move_robots = QSpinBox()
        self._spin_ctrl_move_robots.setMinimum(1)
        self._spin_ctrl_move_robots.setMaximum(ACTION_MOVE_ROBOTS_MAX)
        self._spin_ctrl_move_robots.setValue(1)

        self._button_move_ab = QPushButton("move a to b")
        self._button_move_ba = QPushButton("move b to a")
        self._button_no_move = QPushButton("no move")

        group_transition = QGroupBox("robot transition")
        group_transition_layout = QFormLayout(group_transition)
        group_transition_layout.addRow("robots to move", self._spin_ctrl_move_robots)
        group_transition_layout.addRow(self._button_move_ab)
        group_transition_layout.addRow(self._button_move_ba)
        group_transition_layout.addRow(self._button_no_move)
        layout_simulation.addWidget(group_transition)

        # daily report
        self._edit_report_requests_a = QLineEdit("0")
        self._edit_report_requests_a.setReadOnly(True)
        self._edit_report_requests_b = QLineEdit("0")
        self._edit_report_requests_b.setReadOnly(True)
        self._edit_report_returned_a = QLineEdit("0")
        self._edit_report_returned_a.setReadOnly(True)
        self._edit_report_returned_b = QLineEdit("0")
        self._edit_report_returned_b.setReadOnly(True)
        self._edit_report_moved_robots = QLineEdit("0")
        self._edit_report_moved_robots.setReadOnly(True)
        self._edit_report_reward = QLineEdit("0")
        self._edit_report_reward.setReadOnly(True)

        group_report = QGroupBox("daily report")
        group_report_layout = QFormLayout(group_report)
        group_report_layout.addRow("rent / requested a", self._edit_report_requests_a)
        group_report_layout.addRow("rent / requested b", self._edit_report_requests_b)
        group_report_layout.addRow("returned robots a", self._edit_report_returned_a)
        group_report_layout.addRow("returned robots b", self._edit_report_returned_b)
        group_report_layout.addRow("robots moved", self._edit_report_moved_robots)
        group_report_layout.addRow("reward total", self._edit_report_reward)
        layout_simulation.addWidget(group_report)

        # setup main layout
        layout_main.addWidget(self._canvas)
        layout_main.addLayout(layout_simulation)
        layout_main.addStretch()

        # connect
        self._spin_requests_a.valueChanged.connect(self.plot)
        self._spin_returns_a.valueChanged.connect(self.plot)
        self._spin_requests_b.valueChanged.connect(self.plot)
        self._spin_returns_b.valueChanged.connect(self.plot)

        self._button_move_ab.clicked.connect(self.on_button_move_ab)
        self._button_move_ba.clicked.connect(self.on_button_move_ba)
        self._button_no_move.clicked.connect(self.on_button_no_move)

    def on_button_move_ab(self):
        robots_to_move = self._spin_ctrl_move_robots.value()
        self.perform_action(ACTION_MOVE_ROBOTS_MAX + robots_to_move) 

    def on_button_move_ba(self):
        robots_to_move = self._spin_ctrl_move_robots.value()
        self.perform_action(ACTION_MOVE_ROBOTS_MAX - robots_to_move) 
   
    def on_button_no_move(self):
        self.perform_action(ACTION_MOVE_ROBOTS_MAX)  

    def reset(self):
        self._robots_waiting_a = math.floor(TOTAL_ROBOTS/2)
        self._robots_waiting_b = TOTAL_ROBOTS - self._robots_waiting_a

        self._robots_active_a = 0
        self._robots_active_b = 0

        self._edit_robots_a_waiting.setText(f"{self._robots_waiting_a}")
        self._edit_robots_a_active.setText(f"{self._robots_active_a}")
        self._edit_robots_a_total.setText(f"{self.get_total_robots_a()}")
        self._edit_robots_b_waiting.setText(f"{self._robots_waiting_b}")
        self._edit_robots_b_active.setText(f"{self._robots_active_b}")
        self._edit_robots_b_total.setText(f"{self.get_total_robots_b()}")

        self._edit_report_requests_a.setText("0")
        self._edit_report_requests_b.setText("0")
        self._edit_report_returned_a.setText("0")
        self._edit_report_returned_b.setText("0")
        self._edit_report_moved_robots.setText("0")
        self._edit_report_reward.setText("0")


    def get_total_robots_a(self):
        return self._robots_waiting_a + self._robots_active_a

    def get_total_robots_b(self):
        return self._robots_waiting_b + self._robots_active_b
    
    def get_state(self):
        if True: # state dim 2
            return np.array([self._robots_waiting_a, self._robots_waiting_b], dtype=int)
        else: # state dim 3
            total_a = self.get_total_robots_a()
            total_b = self.get_total_robots_b()
            symmetric_normalized_ratio = (total_a - total_b) / TOTAL_ROBOTS
            return np.array([self._robots_waiting_a, self._robots_waiting_b, symmetric_normalized_ratio], dtype=float)

    def requested_robot_move_ab(self, action:int):
        # move_a_to_b = action - ACTION_MOVE_ROBOTS_MAX
        # 0 - 3 = -3, 3 - 3 = 0. 6 - 3 = 3
        move_ab = action - ACTION_MOVE_ROBOTS_MAX
        robots_a = self.get_total_robots_a()
        robots_b = self.get_total_robots_b()

        if move_ab > 0:
            move_ab = min(move_ab, robots_a)
        elif move_ab < 0: # move b to a
            move_ab = max(move_ab, -robots_b) 
        return move_ab

    def robot_transition(self, action):
        move_ab = self.requested_robot_move_ab(action)

        if move_ab > 0:
            move_ab_waiting = min(move_ab, self._robots_waiting_a)
            move_ab_active = max(move_ab - move_ab_waiting, 0)
            move_ab_active = min(move_ab_active, self._robots_active_a)
        elif move_ab < 0:
            move_ab_waiting = min(abs(move_ab), self._robots_waiting_b)
            move_ab_active = max(abs(move_ab) - move_ab_waiting, 0)
            move_ab_active = min(move_ab_active, self._robots_active_b)
            move_ab_waiting = -move_ab_waiting
            move_ab_active = -move_ab_active
        else:
            move_ab_waiting = 0
            move_ab_active = 0
        
        if True: #  disable possibility to move active robots
            move_ab_active = 0

        self._robots_waiting_a -= move_ab_waiting
        self._robots_waiting_b += move_ab_waiting
        self._robots_active_a -= move_ab_active
        self._robots_active_b += move_ab_active

        moved_robots = move_ab_waiting + move_ab_active
        if moved_robots > 0:
            self._edit_report_moved_robots.setText(f"{moved_robots} a->b")
        elif moved_robots < 0:
            self._edit_report_moved_robots.setText(f"{abs(moved_robots)} b->a")
        else:
            self._edit_report_moved_robots.setText("0")
        
        return abs(moved_robots)
    
    def get_requested_robots_a(self):
        alpha = self._spin_requests_a.value()
        requests = np.random.poisson(alpha)
        return requests

    def get_requested_robots_b(self):
        alpha = self._spin_requests_b.value()
        requests = np.random.poisson(alpha)
        return requests

    def get_returned_robots_a(self):
        alpha = self._spin_returns_a.value()
        returned_robots = np.random.poisson(alpha)
        returned_robots = min(returned_robots, self._robots_active_a)
        return returned_robots

    def get_returned_robots_b(self):
        alpha = self._spin_returns_b.value()
        returned_robots = np.random.poisson(alpha)
        returned_robots = min(returned_robots, self._robots_active_b)
        return returned_robots
    
    def daily_rental(self):
        req_a = self.get_requested_robots_a()
        req_b = self.get_requested_robots_b()

        rent_a = min(self._robots_waiting_a, req_a)
        rent_b = min(self._robots_waiting_b, req_b)

        self._robots_waiting_a -= rent_a
        self._robots_waiting_b -= rent_b
        self._robots_active_a += rent_a
        self._robots_active_b += rent_b

        self._edit_report_requests_a.setText(f"{rent_a} / {req_a}")
        self._edit_report_requests_b.setText(f"{rent_b} / {req_b}")

        return rent_a, rent_b 

    def daily_returns(self):
        returned_a = self.get_returned_robots_a()
        returned_b = self.get_returned_robots_b()

        self._robots_waiting_a += returned_a
        self._robots_waiting_b += returned_b
        self._robots_active_a -= returned_a
        self._robots_active_b -= returned_b

        self._edit_report_returned_a.setText(f"{returned_a}")
        self._edit_report_returned_b.setText(f"{returned_b}")

        return returned_a, returned_b

    def perform_action(self, action:int):
        moved_robots = self.robot_transition(action)
        rent_a, rent_b = self.daily_rental()
        return_a, return_b = self.daily_returns()

        self._edit_robots_a_waiting.setText(f"{self._robots_waiting_a}")
        self._edit_robots_a_active.setText(f"{self._robots_active_a}")
        self._edit_robots_a_total.setText(f"{self.get_total_robots_a()}")
        self._edit_robots_b_waiting.setText(f"{self._robots_waiting_b}")
        self._edit_robots_b_active.setText(f"{self._robots_active_b}")
        self._edit_robots_b_total.setText(f"{self.get_total_robots_b()}")


        reward_a = REWARD_SUCCESSFUL_RENT * rent_a
        reward_b = REWARD_SUCCESSFUL_RENT * rent_b
        reward_move = REWARD_MOVE_ROBOT * moved_robots
        reward = reward_a + reward_b + reward_move

        self._edit_report_reward.setText(f"{reward}")

        return reward

    def get_reward(self):
        return 0.0

    def plot(self):
        self._canvas.plot(
            self._spin_requests_a.value(),
            self._spin_returns_a.value(),
            self._spin_requests_b.value(),
            self._spin_returns_b.value()
        )