import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCommandLineOption, QCommandLineParser

from .experiment import ExperimentDialog
from .factory import environment_factory, agent_factory

def parse_arguments(app):
    parser = QCommandLineParser()
    parser.addHelpOption()
    parser.addVersionOption()

    environment_option = QCommandLineOption(
        ["e", "environment"],
        "Mandatory: This is the name of the environment to show case.",
        "environment"
    )
    parser.addOption(environment_option)

    agent_option = QCommandLineOption(
        ["a", "agent"],
        "Mandatory: This is the name of the agent to show case.",
        "agent"
    )
    parser.addOption(agent_option)

    parser.process(app)

    if not parser.isSet(environment_option) or not parser.isSet(agent_option):
        parser.showHelp(1)

    return parser


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Reinforcement Learning - Demonstration")
    app.setApplicationVersion("1.0")

    
    arg_parser = parse_arguments(app)
    environment = environment_factory(arg_parser)
    agent = agent_factory(arg_parser)

    experiment = ExperimentDialog(environment = environment, agent = agent)
    experiment.show()
    
    return app.exec()

if __name__ == "__main__":
    main()
