import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from enum import IntFlag,  IntEnum

from ...exceptions import LabConfigException

from analytics.probability import poisson_distribution

PLOT_RESOLUTION = 200

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self._figure = Figure(figsize=(5, 4), dpi=100)
        super().__init__(self._figure)

        self.clear()
    
    def clear(self):
        self._figure.clear()
    
    def _prepare_plot(self):
        self.clear()

        self._subplot = self._figure.add_subplot(111)
        self._subplot.set_title("Poisson Distribution")
        self._subplot.set_xlabel(f"daily requested robots")
        self._subplot.set_ylabel(f"probability")

        self._subplot.grid(True)
        
    def _plot_data(self, x, y, label):
        self._subplot.plot(x, y, label=label)
    
    def _calculate_y(self, x, poisson):
        distribution = poisson_distribution(poisson)
        return distribution(x)


    def plot(self, requests_a, returns_a, requests_b, returns_b):
        self._prepare_plot()

        max_poisson = max(requests_a, returns_a, requests_b, returns_b)
        x = np.linspace(0, max_poisson + max_poisson, PLOT_RESOLUTION)

        y = self._calculate_y(x, requests_a)
        self._plot_data(x,y,f"requests a: {requests_a}")
        y = self._calculate_y(x, returns_a)
        self._plot_data(x,y,f"returns a: {returns_a}")

        y = self._calculate_y(x, requests_b)
        self._plot_data(x,y,f"requests b: {requests_b}")
        y = self._calculate_y(x, returns_b)
        self._plot_data(x,y,f"returns b: {returns_b}")

        self._subplot.legend()
        self.draw()
