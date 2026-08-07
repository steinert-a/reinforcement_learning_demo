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

        x = [i for i in range(len(return_log))]
        y = return_log

        if len(x) > 300:
            n = 200
            y_in = np.array(y)
            chunks = np.array_split(y_in, n)
            y = [chunk.mean() for chunk in chunks]
            x = [
                (chunk_indices[0] + chunk_indices[-1]) / 2
                for chunk_indices in np.array_split(np.arange(len(y_in)), n)
            ]

        ax.plot(x,y)
        canvas.draw()


class PolicyHeatmapDialog(QDialog):
    def __init__(self, policy, x_labels, y_labels, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Heat Map")
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        fig = Figure()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)

        im = ax.imshow(
            policy,
            cmap="viridis", 
            aspect="auto"
        )

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        fig.colorbar(im, ax=ax)
        ax.set_title("Policy - Heat Map")

        canvas.draw()