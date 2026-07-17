import sys
import time

from PyQt6.QtWidgets import QDialog, QLineEdit, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QFrame, QProgressDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, callback, steps = 0, sleep_ms = 0):
        super().__init__()
        self._steps = steps
        self._callback = callback
        self._sleep = sleep_ms / 1000.0
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
    
    def run_steps(self):
        for i in range(self._steps):
            if self._cancelled:
                break

            self.call_function()
            self.progress.emit(i + 1)

        self.finished.emit()

    def run_endless(self):
        while not self._cancelled:
            self.call_function()
    
    def run(self):
        if self._steps > 0:
            self.run_steps()
        else:
            self.run_endless()

    def call_function(self):
        self._callback()
        if self._sleep > 0:
            time.sleep(self._sleep)

# https://doc.qt.io/qtforpython-6/tutorials/basictutorial/dialog.html#tutorial-dialog
class ExperimentDialog(QDialog):
    def __init__(self, environment, agent, parent=None):
        super(ExperimentDialog, self).__init__(parent)
        self._environment = environment
        self._agent = agent
        self._step_counter = 0

        self.setup_ui()
        self.reset_all()
    
    def setup_ui(self):
        self.setWindowTitle("Reinforcement Learning - Experiment")

        layout_main = QHBoxLayout(self)
        layout_control = QVBoxLayout()

        group_ctrl_info = QGroupBox("experiment info")
        group_ctrl_info_layout = QFormLayout(group_ctrl_info)
        group_ctrl_info_layout.addRow("version", QLabel("1.0.0"))
        group_ctrl_info_layout.addRow("environment", QLabel(self._environment.name()))
        group_ctrl_info_layout.addRow("agent", QLabel(self._agent.name()))
        layout_control.addWidget(group_ctrl_info)

        group_step_ctrl = QGroupBox("step control")
        group_step_ctrl_layout = QFormLayout(group_step_ctrl)
        self._steps_execute = QSpinBox()
        self._steps_execute.setMinimum(0)
        self._steps_execute.setMaximum(100000)
        self._steps_execute.setValue(1)
        group_step_ctrl_layout.addRow("execute steps (0 endless)", self._steps_execute)
        self._steps_sleep = QSpinBox()
        self._steps_sleep.setMinimum(50)
        self._steps_sleep.setMaximum(60000)
        self._steps_sleep.setValue(100)
        group_step_ctrl_layout.addRow("sleep time [ms] (0 disables)", self._steps_sleep)
        self._steps_done = QLineEdit()
        self._steps_done.setText(f"{self._step_counter}")
        self._steps_done.setReadOnly(True)
        self._steps_done.setEnabled(False)
        group_step_ctrl_layout.addRow("steps done until now", self._steps_done)
        layout_control.addWidget(group_step_ctrl)

        if self._agent.widget() is not None:
            group_agent_ctrl = QGroupBox("agent")
            group_agent_ctrl_layout = QVBoxLayout(group_agent_ctrl)
            group_agent_ctrl_layout.addWidget(self._agent.widget())
            layout_control.addWidget(group_agent_ctrl)

        self._button_reset_all = QPushButton("all")
        self._button_reset_environment = QPushButton("environment")
        self._button_reset_experiment = QPushButton("experiment")
        self._button_reset_agent = QPushButton("agent")
        self._button_execute_steps = QPushButton("execute step(s)")

        group_reset = QGroupBox("reset subsystem")
        group_reset_layout = QHBoxLayout(group_reset)
        # group_reset_layout.addStretch()
        group_reset_layout.addWidget(self._button_reset_experiment)
        group_reset_layout.addWidget(self._button_reset_environment)
        group_reset_layout.addWidget(self._button_reset_agent)
        group_reset_layout.addWidget(self._button_reset_all)
        layout_control.addWidget(group_reset)

        frame_line = QFrame()
        frame_line.setFrameShape(QFrame.Shape.HLine)
        layout_control.addWidget(frame_line)
        layout_control.addWidget(self._button_execute_steps)

        layout_control.addStretch()
        
        layout_main.addLayout(layout_control)

        group_environment = QGroupBox("environment")
        group_environment_layout = QHBoxLayout(group_environment)
        group_environment_layout.addWidget(self._environment.widget())
        layout_main.addWidget(group_environment)

        # connect
        self._button_execute_steps.clicked.connect(self.on_button_execute_steps_clicked)

        self._button_reset_experiment.clicked.connect(self.reset_experiment)
        self._button_reset_environment.clicked.connect(self.reset_environment)
        self._button_reset_agent.clicked.connect(self.reset_agent)
        self._button_reset_all.clicked.connect(self.reset_all)

    def reset_all(self):
        self.reset_environment()
        self.reset_agent()
        self.reset_experiment()

    def reset_environment(self):
        self._environment.reset()

    def reset_agent(self):
        action_space = self._environment.action_space()
        observation, _ = self._environment.state()
        self._agent.reset(observation, action_space)
    
    def reset_experiment(self):
        self._step_counter = 0
        self._steps_done.setText(f"{self._step_counter}")

    
    def execute_step(self):
        observation_0, terminated = self._environment.state()
        if terminated is None or terminated == False:
            action = self._agent.next_action(observation_0)
            observation_1, reward, terminated = self._environment.step(action)
            self._agent.reinforcement_learning(observation_0, action, reward, observation_1, terminated)
        self._step_counter += 1
        self._steps_done.setText(f"{self._step_counter}")

    def on_button_execute_steps_clicked(self):
        steps_to_do = self._steps_execute.value()
        if steps_to_do == 1:
            self.execute_step()
        else:
            steps_sleep_ms = self._steps_sleep.value()

            progress_dialog = QProgressDialog(
                "Processing...",
                "Cancel",
                0,
                steps_to_do,
                self,
            )
            progress_dialog.setWindowTitle("Progress")
            progress_dialog.setValue(0)

            worker = Worker(self.execute_step, steps_to_do, steps_sleep_ms)

            worker.progress.connect(
                progress_dialog.setValue
            )
            worker.finished.connect(
                progress_dialog.close
            )

            progress_dialog.canceled.connect(
                worker.cancel
            )

            
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            worker.start()
            progress_dialog.show()

