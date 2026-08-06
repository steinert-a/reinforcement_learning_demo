from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout,
    QPushButton, QMainWindow
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

class LogEpisodeReturnDialog(QDialog):
    def __init__(self, return_log, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        fig = Figure()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.set_title("Episode Return Log")
        ax.set_xlabel(f"episode")
        ax.set_ylabel(f"return")

        ax.plot([i for i in range(len(return_log))],return_log)
        canvas.draw()


class ActionValuesHeatmapDialog(QDialog):
    def __init__(self, action_values, x_labels, y_labels, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Heat Map")
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        fig = Figure()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)

        im = ax.imshow(
            action_values,
            cmap="viridis", 
            aspect="auto"
        )

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        fig.colorbar(im, ax=ax)
        ax.set_title("Action Values - Heat Map")

        canvas.draw()