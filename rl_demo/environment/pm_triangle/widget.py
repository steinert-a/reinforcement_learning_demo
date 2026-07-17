from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QGroupBox, QFormLayout, QCheckBox, QLineEdit
from PyQt6.QtCore import Qt

from .plotter import MplCanvas, PlotTarget, PlotAxis
from .simulation import *

from enum import StrEnum

ACTION_STEP_SIZE = 1

class Actions(StrEnum):
    ACTION_NONE = "none"
    ACTION_OVERRIDE_DOWN = "override_down"
    ACTION_OVERRIDE_UP = "override_up"
    ACTION_COOLING_DOWN = "cooling_down"
    ACTION_COOLING_UP = "cooling_up"


class PmTriangleWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._plot_target = 0

        self.setup_ui()
        self.calculate_state()
        self.plot()
    
    def setup_ui(self):
        self.setWindowTitle("Reinforcement Learning - Experiment")

        layout_main = QHBoxLayout(self)
        layout_control = QVBoxLayout()
        layout_simulation= QVBoxLayout()

        # control
        self._canvas = MplCanvas()
        layout_control.addWidget(self._canvas)

        self._slider_quality = QSlider(Qt.Orientation.Horizontal)
        self._slider_quality.setRange(0, 100)
        self._slider_quality.setValue(50)
        self._slider_time = QSlider(Qt.Orientation.Horizontal)
        self._slider_time.setRange(0, 100)
        self._slider_time.setValue(50)
        self._slider_coast = QSlider(Qt.Orientation.Horizontal)
        self._slider_coast.setRange(0, 100)
        self._slider_coast.setValue(50)

        group_triangle = QGroupBox("project management triangle")
        group_triangle_layout = QFormLayout(group_triangle)
        group_triangle_layout.addRow("quality", self._slider_quality)
        group_triangle_layout.addRow("coast", self._slider_coast)
        group_triangle_layout.addRow("time", self._slider_time)
        layout_control.addWidget(group_triangle)

        # simulation
        self._check_temperature = QCheckBox("temperature")
        self._edit_temperature = QLineEdit("")
        self._edit_temperature.setEnabled(False)

        self._check_vibration = QCheckBox("vibration")
        self._edit_vibration = QLineEdit("")
        self._edit_vibration.setEnabled(False)

        group_quality = QGroupBox("quality")
        group_quality_layout = QFormLayout(group_quality)
        group_quality_layout.addRow(self._check_temperature, self._edit_temperature)
        group_quality_layout.addRow(self._check_vibration, self._edit_vibration)
        layout_simulation.addWidget(group_quality)

        self._check_energy = QCheckBox("energy")
        self._edit_energy = QLineEdit("")
        self._edit_energy.setEnabled(False)

        self._check_wearing = QCheckBox("tool wear")
        self._edit_wearing = QLineEdit("")
        self._edit_wearing.setEnabled(False)

        group_coast = QGroupBox("coast")
        group_coast_layout = QFormLayout(group_coast)
        group_coast_layout.addRow(self._check_energy, self._edit_energy)
        group_coast_layout.addRow(self._check_wearing, self._edit_wearing)
        layout_simulation.addWidget(group_coast)

    
        self._check_override = QCheckBox("override")
        self._edit_override = QLineEdit("")
        self._edit_override.setEnabled(False)

        self._check_junk = QCheckBox("junk ratio")
        self._edit_junk = QLineEdit("")
        self._edit_junk.setEnabled(False)

        group_time = QGroupBox("time")
        group_time_layout = QFormLayout(group_time)
        group_time_layout.addRow(self._check_override, self._edit_override)
        group_time_layout.addRow(self._check_junk, self._edit_junk)
        layout_simulation.addWidget(group_time)

        self._check_state_override = QCheckBox("override")
        self._slider_state_override = QSlider(Qt.Orientation.Horizontal)
        self._slider_state_override.setRange(0, 100)
        self._slider_state_override.setValue(50)

        self._check_state_cooling = QCheckBox("cooling")
        self._slider_state_cooling = QSlider(Qt.Orientation.Horizontal)
        self._slider_state_cooling.setRange(0, 100)
        self._slider_state_cooling.setValue(50)

        group_state = QGroupBox("state")
        group_state_layout = QFormLayout(group_state)
        group_state_layout.addRow(self._check_state_override, self._slider_state_override)
        group_state_layout.addRow(self._check_state_cooling , self._slider_state_cooling)
        layout_simulation.addWidget(group_state)

        self._check_reward = QCheckBox("reward")
        self._edit_reward = QLineEdit("")
        self._edit_reward.setEnabled(False)

        group_reward = QGroupBox("reward")
        group_reward_layout = QFormLayout(group_reward)
        group_reward_layout.addRow(self._check_reward, self._edit_reward)
        layout_simulation.addWidget(group_reward)

        layout_simulation.addStretch()
        layout_main.addLayout(layout_control)
        layout_main.addLayout(layout_simulation)

        # connect
        self._slider_state_override.valueChanged.connect(self.calculate_state)
        self._slider_state_cooling.valueChanged.connect(self.calculate_state)

        self._slider_state_override.valueChanged.connect(self.plot_if_not_3d)
        self._slider_state_cooling.valueChanged.connect(self.plot_if_not_3d)

        self._check_state_override.toggled.connect(self.plot)
        self._check_state_cooling.toggled.connect(self.plot)

        self._check_temperature.toggled.connect(self.plot_temperature)
        self._check_vibration.toggled.connect(self.plot_vibration)

        self._check_energy.toggled.connect(self.plot_energy)
        self._check_wearing.toggled.connect(self.plot_tool_wear)

        self._check_override.toggled.connect(self.plot_override)
        self._check_junk.toggled.connect(self.plot_junk)

        self._check_reward.toggled.connect(self.plot)

        self._slider_coast.valueChanged.connect(self.plot_if_default)
        self._slider_time.valueChanged.connect(self.plot_if_default)
        self._slider_quality.valueChanged.connect(self.plot_if_default)

        self._slider_coast.valueChanged.connect(self.calculate_state)
        self._slider_time.valueChanged.connect(self.calculate_state)
        self._slider_quality.valueChanged.connect(self.calculate_state)
    
    def reset(self):
        # self._slider_coast.setValue(50)
        # self._slider_time.setValue(50)
        # self._slider_quality.setValue(50)

        self._slider_state_override.setValue(50)
        self._slider_state_cooling.setValue(50)

    def get_weight_coast(self):
        value = self._slider_coast.value()
        return value/100.0
    
    def get_weight_time(self):
        value = self._slider_time.value()
        return value/100.0
    
    def get_weight_quality(self):
        value = self._slider_quality.value()
        return value/100.0
    
    def get_override(self):
        value = self._slider_state_override.value()
        value_percent = value/100.0
        return calc_override(value_percent)

    def get_state(self):
        value_cooling = self._slider_state_cooling.value()
        value_cooling_percent = value_cooling/100.0

        value_override = self._slider_state_override.value()
        value_override_percent = value_override/100.0

        return np.array([value_override_percent, value_cooling_percent], dtype=float)
    
    def perform_action(self, action:int):
        next_action = list(Actions)[action]
        if next_action == Actions.ACTION_OVERRIDE_DOWN:
            value = self._slider_state_override.value()
            next_value = value - ACTION_STEP_SIZE
            next_value = max(0, next_value)
            self._slider_state_override.setValue(next_value)
        if next_action == Actions.ACTION_OVERRIDE_UP:
            value = self._slider_state_override.value()
            next_value = value + ACTION_STEP_SIZE
            next_value = min(100, next_value)
            self._slider_state_override.setValue(next_value)
        if next_action == Actions.ACTION_COOLING_DOWN:
            value = self._slider_state_cooling.value()
            next_value = value - ACTION_STEP_SIZE
            next_value = max(0, next_value)
            self._slider_state_cooling.setValue(next_value)
        if next_action == Actions.ACTION_COOLING_UP:
            value = self._slider_state_cooling.value()
            next_value = value + ACTION_STEP_SIZE
            next_value = min(100, next_value)
            self._slider_state_cooling.setValue(next_value)

    def get_reward(self):
        override = self.get_override()
        cooling_power = self.get_cooling_power()
        reward = calc_reward(
                self.get_weight_coast(),
                self.get_weight_time(),
                self.get_weight_quality(),
                override, cooling_power
            )
        return reward
    
    def get_cooling_power(self):
        value = self._slider_state_cooling.value()
        value_percent = value/100.0
        return calc_cooling_power(value_percent)
    
    def get_axis(self):
        if self._check_state_override.isChecked() and not self._check_state_cooling.isChecked():
            return PlotAxis.PLOT_AXIS_OVERRIDE
        if not self._check_state_override.isChecked() and self._check_state_cooling.isChecked():
            return PlotAxis.PLOT_AXIS_COOLING
        return PlotAxis.PLOT_AXIS_3D

    def calculate_state(self):
        override = self.get_override()
        cooling_power = self.get_cooling_power()
        temperature = calc_temperature(override, cooling_power)
        tool_wear = calc_tool_wear(override, cooling_power)
        vibration = calc_vibration(override, cooling_power)
        energy = calc_energy(override, cooling_power)
        scrap_ratio = calc_scrap_ratio(override, cooling_power)
        reward = calc_reward(
                self.get_weight_coast(),
                self.get_weight_time(),
                self.get_weight_quality(),
                override, cooling_power
            )

        self._edit_override.setText(f"{override:.02f} {UNIT_OVERRIDE}")
        self._edit_temperature.setText(f"{temperature:.02f} {UNIT_TEMPERATURE}")
        self._edit_wearing.setText(f"{tool_wear:.02f} {UNIT_TOOL_WEAR}")
        self._edit_vibration.setText(f"{vibration:.02f} {UNIT_VIBRATION}")
        self._edit_energy.setText(f"{energy:.02f} {UNIT_ENERGY}")
        self._edit_junk.setText(f"{scrap_ratio:.02f} {UNIT_SCRAP_RATIO}")
        self._edit_reward.setText(f"{reward:.02f}")

    def plot_temperature(self, checked: bool):
        if checked:
            self._plot_target |= PlotTarget.FLAG_PLOT_TEMPERATURE
        else:
            self._plot_target &= ~PlotTarget.FLAG_PLOT_TEMPERATURE

        self.plot()

    def plot_vibration(self, checked: bool):
        if checked:
            self._plot_target |= PlotTarget.FLAG_PLOT_VIBRATION
        else:
            self._plot_target &= ~PlotTarget.FLAG_PLOT_VIBRATION

        self.plot()

    def plot_energy(self, checked: bool):
        if checked:
            self._plot_target |= PlotTarget.FLAG_PLOT_ENERGY
        else:
            self._plot_target &= ~PlotTarget.FLAG_PLOT_ENERGY

        self.plot()

    def plot_tool_wear(self, checked: bool):
        if checked:
            self._plot_target |= PlotTarget.FLAG_PLOT_TOOL_WEAR
        else:
            self._plot_target &= ~PlotTarget.FLAG_PLOT_TOOL_WEAR

        self.plot()

    def plot_override(self, checked: bool):
        if checked:
            self._plot_target |= PlotTarget.FLAG_PLOT_OVERRIDE
        else:
            self._plot_target &= ~PlotTarget.FLAG_PLOT_OVERRIDE

        self.plot()

    def plot_junk(self, checked: bool):
        if checked:
            self._plot_target |= PlotTarget.FLAG_PLOT_SCRAP_RATIO
        else:
            self._plot_target &= ~PlotTarget.FLAG_PLOT_SCRAP_RATIO

        self.plot()

    def plot_if_not_3d(self):
        if self.get_axis() != PlotAxis.PLOT_AXIS_3D:
            self.plot()

    def plot_if_default(self):
        if self._plot_target == 0:
            self.plot()

    def plot(self):
        self._canvas.plot(
            self.get_override(),
            self.get_cooling_power(),
            self._plot_target,
            self.get_axis(),
            self._check_reward.isChecked(),
            lambda o,c: calc_reward(
                self.get_weight_coast(),
                self.get_weight_time(),
                self.get_weight_quality(),
                o,c
            ),
            "reward total"
        )
