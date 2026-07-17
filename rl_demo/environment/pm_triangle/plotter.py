import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from enum import IntFlag,  IntEnum

from .simulation import *
from ...exceptions import LabConfigException


PLOT_RESOLUTION = 200

class PlotTarget(IntFlag):
    FLAG_PLOT_TEMPERATURE = 1 << 0
    FLAG_PLOT_VIBRATION = 1 << 1
    FLAG_PLOT_TOOL_WEAR = 1 << 2
    FLAG_PLOT_ENERGY = 1 << 3
    FLAG_PLOT_OVERRIDE = 1 << 4
    FLAG_PLOT_SCRAP_RATIO = 1 << 5

class PlotAxis(IntEnum):
    PLOT_AXIS_3D = 0
    PLOT_AXIS_OVERRIDE = 1
    PLOT_AXIS_COOLING = 2



class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self._figure = Figure(figsize=(5, 4), dpi=100)
        super().__init__(self._figure)

        self.clear()
    
    def clear(self, used_axes = PlotAxis.PLOT_AXIS_3D):
        self._figure.clear()
        self._used_axes = used_axes
        self._subplot = None
    
    def _prepare_plot(self, used_axes):
        self.clear(used_axes)

        if used_axes == PlotAxis.PLOT_AXIS_3D:
            self._subplot = self._figure.add_subplot(111, projection='3d')
            self._subplot.set_title("experiment simulation 3D")
        else:
            self._subplot = self._figure.add_subplot(111)
            self._subplot.set_title("experiment simulation 2D")

            if used_axes == PlotAxis.PLOT_AXIS_OVERRIDE:
                self._subplot.set_xlabel(f"override [{UNIT_OVERRIDE}]")
            elif used_axes == PlotAxis.PLOT_AXIS_COOLING:
                self._subplot.set_xlabel(f"cooling power [{UNIT_COOLING_POWER}]")

            self._subplot.grid(True)

    def _calc_function_input(self):
        x = np.linspace(MIN_OVERRIDE, MAX_OVERRIDE, PLOT_RESOLUTION)
        y = np.linspace(MIN_COOLING_POWER, MAX_COOLING_POWER, PLOT_RESOLUTION)

        if self._used_axes == PlotAxis.PLOT_AXIS_3D:
            return np.meshgrid(x, y)
        else:
            if self._used_axes == PlotAxis.PLOT_AXIS_OVERRIDE:
                return x, None
            elif self._used_axes == PlotAxis.PLOT_AXIS_COOLING:
                return y, None
            else:
                raise LabConfigException(f"invalid axis configuration {self._used_axes}")
        
    def _calc_function_output(self, x, y, f, override, cooling_power):
        if self._used_axes == PlotAxis.PLOT_AXIS_3D:
            return f(x,y)
        else:
            if self._used_axes == PlotAxis.PLOT_AXIS_OVERRIDE:
                return f(x,cooling_power)
            elif self._used_axes == PlotAxis.PLOT_AXIS_COOLING:
                return f(override,x)

    def _plot_data(self, x, y, z, label):
        if self._used_axes == PlotAxis.PLOT_AXIS_3D:
            self._subplot.plot_surface(x,y,z, cmap="viridis", label=label)
        else:
            self._subplot.plot(x, z, label=label)

    def plot(self, override, cooling_power, target: PlotTarget, used_axes:PlotAxis, reward, default_function = None, default_label = None):
        self._prepare_plot(used_axes)

        x,y = self._calc_function_input()
        if target & PlotTarget.FLAG_PLOT_TEMPERATURE:
            z = self._calc_function_output(x,y, calc_reward_temperature if reward else calc_temperature, override, cooling_power)
            self._plot_data(x,y,z,"temperature reward" if reward else f"temperature [{UNIT_TEMPERATURE}]")
        if target & PlotTarget.FLAG_PLOT_VIBRATION:
            z = self._calc_function_output(x,y, calc_reward_vibration if reward else calc_vibration, override, cooling_power)
            self._plot_data(x,y,z,"vibration reward" if reward else f"vibration [{UNIT_VIBRATION}]")

        if target & PlotTarget.FLAG_PLOT_ENERGY:
            z = self._calc_function_output(x,y, calc_reward_energy if reward else calc_energy, override, cooling_power)
            self._plot_data(x,y,z,"energy reward" if reward else f"energy [{UNIT_ENERGY}]")
        if target & PlotTarget.FLAG_PLOT_TOOL_WEAR:
            z = self._calc_function_output(x,y, calc_reward_tool_wear if reward else calc_tool_wear, override, cooling_power)
            self._plot_data(x,y,z,"tool wear reward" if reward else f"tool wear [{UNIT_TOOL_WEAR}]")

        if target & PlotTarget.FLAG_PLOT_OVERRIDE:
            z = self._calc_function_output(x,y, calc_reward_override2 if reward else calc_override2, override, cooling_power)
            self._plot_data(x,y,z,"override reward" if reward else f"override [{UNIT_OVERRIDE}]")
        if target & PlotTarget.FLAG_PLOT_SCRAP_RATIO:
            z = self._calc_function_output(x,y, calc_reward_scrap_ratio if reward else calc_scrap_ratio, override, cooling_power)
            self._plot_data(x,y,z,"scrap ratio reward" if reward else f"scrap ratio [{UNIT_SCRAP_RATIO}]")

        if target == 0 and default_function is not None:
            z = self._calc_function_output(x,y, default_function, override, cooling_power)
            self._plot_data(x,y,z,default_label)

        self._subplot.legend()
        self.draw()
