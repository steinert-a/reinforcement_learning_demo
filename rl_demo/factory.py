from .exceptions import LabCommandLineException

def environment_factory(args):
    environment = args.value("environment")
    
    match environment:
        case "triangle":
            from .environment.pm_triangle import EnvPmTriangle
            return EnvPmTriangle(args)
        case "rental":
            from .environment.robot_rental import EnvRobotRental
            return EnvRobotRental(args)

    raise LabCommandLineException(f"environment {environment} does not exist")

def agent_factory(args):
    agent = args.value("agent")
    
    match agent:
        case "sample":
            from .agent.sample_average import AgentSampleAverage
            return AgentSampleAverage(args)
        case "weighted":
            from .agent.weighted_average import AgentWeightedAverage
            return AgentWeightedAverage(args)
        case "gradient":
            from .agent.gradient import AgentGradient
            return AgentGradient(args)
        case "gradient_nn":
            from .agent.gradient_nn import AgentGradientNn
            return AgentGradientNn(args)
        case "policy_iter":
            from .agent.policy_iteration import AgentPolicyIteration
            return AgentPolicyIteration(args)
        case "mc_control":
            from .agent.mc_control import AgentMonteCarloControl
            return AgentMonteCarloControl(args)
        case "sarsa":
            from .agent.sarsa import AgentTdSarsa
            return AgentTdSarsa(args)
        case "q_learning":
            from .agent.q_learning import AgentQLearning
            return AgentQLearning(args)

    raise LabCommandLineException(f"agent {agent} does not exist")