from .environment.pm_triangle import EnvPmTriangle

from .agent.sample_average import AgentSampleAverage
from .agent.weighted_average import AgentWeightedAverage
from .agent.gradient import AgentGradient
from .agent.gradient_nn import AgentGradientNn

from .exceptions import LabCommandLineException

def environment_factory(args):
    environment = args.value("environment")
    
    match environment:
        case "triangle":
            return EnvPmTriangle(args)

    raise LabCommandLineException(f"environment {environment} does not exist")

def agent_factory(args):
    agent = args.value("agent")
    
    match agent:
        case "sample":
            return AgentSampleAverage(args)
        case "weighted":
            return AgentWeightedAverage(args)
        case "gradient":
            return AgentGradient(args)
        case "gradient_nn":
            return AgentGradientNn(args)

    raise LabCommandLineException(f"agent {agent} does not exist")