# Gradient Bandit Algorithm as Stochastic Gradient Ascent

The gradient bandit algorithm can be viewed as a special case of policy-gradient reinforcement learning.

We define a preference value $H_t(a)$ for each action $a$, and the policy is given by a softmax:

$$
\pi_t(a)
=
\frac{\exp(H_t(a))}
     {\sum_b \exp(H_t(b))}
$$

The objective is the expected reward

$$
J(H) = E[R_t].
$$

Since the reward depends on the selected action,

$$
J(H)
=
\sum_b \pi_t(b) q(b),
$$

where

$$
q(b)
=
E[R_t \mid A_t=b].
$$

---

## Step 1: Differentiate the Objective

We want

$$
\frac{\partial J}{\partial H_t(a)}
=
\frac{\partial E[R_t]}{\partial H_t(a)}.
$$

Applying the chain rule:

$$
\frac{\partial J}{\partial H_t(a)}
=
\sum_b q(b)
\frac{\partial \pi_t(b)}
     {\partial H_t(a)}.
$$

---

## Step 2: Derivative of the Softmax

For the softmax policy,

$$
\frac{\partial \pi_t(b)}
     {\partial H_t(a)}
=
\pi_t(b)
\left(
\mathbf{1}_{a=b}
-
\pi_t(a)
\right),
$$

where

$$
\mathbf{1}_{a=b}
=
\begin{cases}
1 & a=b \\
0 & a\neq b
\end{cases}
$$

is the indicator function.

---

## Step 3: Substitute into the Gradient

Substituting the softmax derivative:

$$
\frac{\partial J}
     {\partial H_t(a)}
=
\sum_b
q(b)\,
\pi_t(b)
\left(
\mathbf{1}_{a=b}
-
\pi_t(a)
\right).
$$

Splitting the sum:

$$
=
q(a)\pi_t(a)
-
\pi_t(a)
\sum_b \pi_t(b)q(b).
$$

But

$$
\sum_b \pi_t(b)q(b)
=
E[R_t].
$$

Therefore

$$
\boxed{
\frac{\partial E[R_t]}
     {\partial H_t(a)}
=
\pi_t(a)
\left(
q(a)-E[R_t]
\right)
}
$$

This is the exact gradient of the expected reward with respect to the preference $H_t(a)$.

---

## Stochastic Gradient Estimate

The true action value

$$
q(a)
=
E[R_t|A_t=a]
$$

is unknown.

Instead, after executing an action $A_t$ and receiving a reward $R_t$, we use the sampled reward as an unbiased estimate.

Using the identity

$$
\frac{\partial \log \pi_t(A_t)}
     {\partial H_t(a)}
=
\mathbf{1}_{A_t=a}
-
\pi_t(a),
$$

the gradient estimate becomes

$$
R_t
\left(
\mathbf{1}_{A_t=a}
-
\pi_t(a)
\right).
$$

This yields the update rule

$$
\boxed{
H_{t+1}(a)
=
H_t(a)
+
\alpha R_t
\left(
\mathbf{1}_{A_t=a}
-
\pi_t(a)
\right)
}
$$

which is stochastic gradient ascent on the expected reward.

---

## Adding a Baseline

A baseline $B_t$ can be subtracted without changing the expected gradient:

$$
\boxed{
H_{t+1}(a)
=
H_t(a)
+
\alpha
(R_t-B_t)
\left(
\mathbf{1}_{A_t=a}
-
\pi_t(a)
\right)
}
$$

A common choice is the running average reward

$$
B_t=\bar R_t.
$$

This reduces variance and improves learning stability.

---

## Connection to Policy Gradients

The gradient bandit algorithm is the simplest instance of the REINFORCE policy-gradient method:

$$
\nabla J
=
E
\left[
R_t
\nabla \log \pi_t(A_t)
\right].
$$

The only difference is that in the gradient bandit setting the policy parameters are the action preferences $H_t(a)$, whereas in modern RL these preferences are usually produced by a neural network.

# Neural Network Generalization of the Gradient Bandit Algorithm

Yes, absolutely.

In fact, replacing the preference table $H(a)$ with a neural network is the natural step from the gradient bandit algorithm to modern policy-gradient methods such as REINFORCE.

---

## Original Gradient Bandit

In the gradient bandit algorithm, each action has a preference:

$$
H(a)
$$

and the policy is obtained via a softmax:

$$
\pi(a)
=
\frac{\exp(H(a))}
     {\sum_b \exp(H(b))}.
$$

Here, $H(a)$ is simply a learned scalar for each action.

For three actions $a,b,c$, we could write

$$
H=
\begin{bmatrix}
H_a\\
H_b\\
H_c
\end{bmatrix}.
$$

These preferences are independent of any state.

---

## Replacing the Preference Table with a Neural Network

Suppose the environment has a state $s$.

Instead of storing fixed preferences, let a neural network compute them:

$$
H(s;\theta)
=
\begin{bmatrix}
H_a(s;\theta)\\
H_b(s;\theta)\\
H_c(s;\theta)
\end{bmatrix},
$$

where

- $s$ is the state,
- $\theta$ are the network parameters,
- $H_a,H_b,H_c$ are the outputs of the network.

The outputs are often called **logits** or **preferences**.

A simple architecture would look like:

```text
State s
   │
   ▼
Perceptron / Neural Network
   │
   ├── H_a(s)
   ├── H_b(s)
   └── H_c(s)
```

---

## From Preferences to Action Probabilities

The logits are converted into a probability distribution using a softmax:

$$
\pi(a|s)
=
\frac{\exp(H_a(s))}
     {\exp(H_a(s))
      +
      \exp(H_b(s))
      +
      \exp(H_c(s))}
$$

and similarly

$$
\pi(b|s)
=
\frac{\exp(H_b(s))}
     {\sum_i \exp(H_i(s))}
$$

$$
\pi(c|s)
=
\frac{\exp(H_c(s))}
     {\sum_i \exp(H_i(s))}.
$$

The resulting vector

$$
\pi(\cdot|s)
=
\begin{bmatrix}
\pi(a|s)\\
\pi(b|s)\\
\pi(c|s)
\end{bmatrix}
$$

represents the stochastic policy.

---

## Interpretation

The original gradient bandit learns

$$
H(a)
$$

for each action independently.

The neural-network version learns

$$
H(s,a;\theta),
$$

or equivalently

$$
H(s;\theta)
=
(H_a,H_b,H_c).
$$

The important difference is that the preferences now depend on the current state.

This allows the agent to:

- choose different actions in different states,
- generalize between similar states,
- handle large state spaces,
- handle continuous state spaces.

---

## Connection to REINFORCE

Once the preferences are produced by a neural network, we no longer directly update $H_a,H_b,H_c$.

Instead we update the network parameters $\theta$.

The policy-gradient objective becomes

$$
J(\theta)
=
E[R].
$$

The REINFORCE gradient is

$$
\nabla_\theta J(\theta)
=
E
\left[
(R-B)
\nabla_\theta
\log \pi(A|s;\theta)
\right].
$$

A common training loss is

$$
L(\theta)
=
-(R-B)
\log \pi(A|s;\theta).
$$

Backpropagation automatically computes

$$
\nabla_\theta L,
$$

which propagates through

$$
\theta
\rightarrow
H(s;\theta)
\rightarrow
\pi(\cdot|s)
\rightarrow
\log\pi(A|s).
$$

---

## Relation to the Gradient Bandit Algorithm

The gradient bandit algorithm can be viewed as a special case of this neural-network formulation:

- only one state exists,
- the network has no hidden layers,
- the outputs are just trainable bias terms,
- the outputs correspond directly to $H(a)$.

In that sense,

$$
H(a)
$$

is simply the simplest possible policy network.

Replacing it with

$$
H(s;\theta)
=
(H_a,H_b,H_c)
$$

is exactly how modern policy-gradient methods extend the gradient bandit approach to state-dependent decision making.

---

## Summary

Your proposed model

$$
s
\;\longrightarrow\;
\text{Perceptron}
\;\longrightarrow\;
(H_a,H_b,H_c)
\;\longrightarrow\;
\text{Softmax}
\;\longrightarrow\;
\pi(a|s)
$$

is completely valid and is essentially a **policy network**.

The original gradient bandit is simply the special case where the network has no state input and the outputs $H_a,H_b,H_c$ are learned directly.


---

## Unbiasedness in the REINFORCE Setting

The usual definition of an unbiased estimator is:

$$
E[\hat{\theta}] = \theta
$$

or equivalently

$$
E[\hat{\theta}-\theta]=0.
$$

This assumes that $\theta$ is a **constant parameter**.

---

### Case 1: Estimating $q(a)$

For a fixed action $a$,

$$
q(a)=E[R_t|A_t=a]
$$

is a constant.

Therefore $R_t$ is an unbiased estimator of $q(a)$ because

$$
E[R_t|A_t=a]
=
q(a).
$$

Equivalently,

$$
E[R_t-q(a)\mid A_t=a]
=
0
$$

This is the standard notion of unbiasedness.

---

### Case 2: Estimating $q(A_t)$

Before observing the action,

$$
A_t \sim \pi
$$

is a random variable.

Therefore

$$
q(A_t)
$$

is also a random variable:

$$
q(A_t)
=
\begin{cases}
q(a) & \text{if } A_t=a\\
q(b) & \text{if } A_t=b\\
q(c) & \text{if } A_t=c
\end{cases}
$$

In this case, the correct statement is

$$
E[R_t\mid A_t]
=
q(A_t)
$$

or equivalently,

$$
E[R_t-q(A_t)\mid A_t]
=
0
$$

This means that $R_t$ is conditionally unbiased for the random variable $q(A_t)$.

### A Related True Statement

Using the tower property,

$$
E[q(A_t)]
=
E(E[R_t\mid A_t])
=
E[R_t].
$$

Therefore,

$$
\boxed{
E[q(A_t)]
=
E[R_t]
}
$$

---

### Key Identity Used by REINFORCE

The most important identity is

$$
\boxed{
E[R_t\mid A_t]
=
q(A_t).
}
$$

Equivalently,

$$
\boxed{
E[R_t-q(A_t)\mid A_t]
=
0.
}
$$

This is why the sampled reward $R_t$ can be used as a Monte Carlo replacement for the unknown quantity $q(A_t)$ in policy-gradient methods such as REINFORCE.


# links
* [Estimator](https://en.wikipedia.org/wiki/Estimator)