# Reinforcement Learning Demo
This repository contains a collection of reinforcement learning agents and environments. Its goal is to explore and evaluate different agent implementations across a variety of simulated environments.


## Agents
The following list gives an overview of the implemented RL agents, along with additional information.

### Sample Average [sample]
The Sample Average agent is one of the simplest reinforcement learning agents. It keeps track of the rewards received for each action and updates the estimated value of an action by calculating the average of all rewards obtained from that action.

### Weighted Average [weighted]
The Weighted Average agent is similar to the Sample Average agent, but it updates the estimated action values using a weighted average of the observed rewards.

### Gradient Ascent [gradient]
The Gradient Ascent method seeks to maximize the expected reward, $E[R_t]$. The policy is parameterized by action preferences, which are transformed into a probability distribution over actions using the Softmax function.

### Gradient Ascent - Neural Network [gradient_nn]
The Gradient Ascent Neural Network method extends the Gradient Ascent approach by using a perceptron to learn and represent the action preferences. These learned preferences are then used to derive the policy.

### Dynamic Programming - Policy Iteration [policy_iter]
Policy Iteration is a Dynamic Programming algorithm and therefore requires knowledge of the environment's dynamics. While some simulated environments can provide this information, this RL demo repository does not cover Dynamic Programming algorithms.

For this reason, the environment dynamics used here are assumed to be deterministic. In other words, for any given situation, executing a specific action always leads to the same resulting state and produces the same reward.

### Monte Carlo Control [mc_control]
Monte Carlo Control is implemented here as an on-policy method with an $\epsilon$-soft policy.

### Temporal-Difference Learning - SARSA - On Policy [sarsa]
The implemented Temporal-Difference learning method, State-Action-Reward-State-Action (SARSA), uses an on-policy $\epsilon$-greedy strategy.

## Environments
The following list provides an overview of the implemented environments, along with some additional information.

### Project Management Triangle [triangle]
The Project Management Triangle (good, fast, cheap) illustrates the trade-offs between three competing project objectives: quality, speed, and cost. Improving one aspect typically has an impact on the others. For example, accelerating a project often requires compromises in quality or increased costs. Prioritizing high quality generally results in longer delivery times and higher expenses. Conversely, reducing costs may require lowering quality expectations or extending the project timeline. The concept highlights that it is difficult to optimize all three factors simultaneously, and project teams must balance them according to their priorities. 

The PM Triangle environment simulates different machine signals, with two signals for each objective. Based on these signals, a reward is calculated for each objective. The user can specify the importance of each objective by assigning weights. The weighted sum of the rewards is then used as input for the agent.

The agent can perform five actions: increase or decrease the manufacturing speed, increase or decrease the cooling fluid flow, or do nothing.

### Robot Rental [rental]
An agent should learn how many robots to move between two robot locations each night in order to maximize long-term profit. This problem is known as **Jack's Car Rental** from Sutton & Barto's *Reinforcement Learning: An Introduction*.

## Links
* [Reinforcement Learning, second edition: An Introduction](https://www.amazon.de/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262039249?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=2JQH61FI6BNU4&dib=eyJ2IjoiMSJ9.PgzjIs-Np-ssAU8N3sfGxf33cSZbQEDbfzQdG7mdDqVjNjBo-hWm-rq3T6KG-uj7VlJQ3kcl1jUdMkjocvl00fFaTE21c7A88mRTsOvWovwgml-u_zQyDWykuTGaCXhWRTDYAUUQDeDfUwNvlncsbputs9AX53c0YUfnuSmorL_PSmlWrZoLCl662diDTdrNc6ISlCPWQrcpJAfHPkY36IU3ZZBwN1-J83jOhFUfOs4.un1pDYfuLZGffQ85OixrSWsATbAh9sOugQGky5rcu0Q&dib_tag=se&keywords=reinforcement+learning&qid=1768468727&sprefix=reinforcement+learnin%2Caps%2C108&sr=8-1) 