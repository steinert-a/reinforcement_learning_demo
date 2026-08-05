# Reinforcement Learning: Car Rental Problem

## Goal of the Problem

An agent should learn how many cars to move between two rental locations each night in order to maximize long-term profit.

This problem is known as **Jack's Car Rental** from Sutton & Barto's *Reinforcement Learning: An Introduction*.

---

# Environment

## State

The state describes the number of available cars at both rental locations.

```python
state = (cars_A, cars_B)
```

Example:

```python
state = (10, 15)
```

Meaning:

- 10 cars at location A
- 15 cars at location B

Assuming a maximum capacity of 20 cars per location:

```python
MAX_CARS = 20
```

the total number of states is:

```python
21 * 21 = 441
```

---

## Action

Every night, the agent decides how many cars to transfer between locations.

```python
action in {-5, -4, ..., 4, 5}
```

Interpretation:

```text
+3 -> Move 3 cars from A to B
-2 -> Move 2 cars from B to A
```

The maximum number of transferable cars per night is 5.

---

## Demand and Returns

Rental requests follow Poisson distributions:

```python
requests_A ~ Poisson(3)
requests_B ~ Poisson(4)
```

Returns are also modeled using Poisson distributions:

```python
returns_A ~ Poisson(3)
returns_B ~ Poisson(2)
```

These parameters originate from the classical Sutton & Barto example.

---

## Reward

For each successful rental:

```python
+10 $
```

For each transferred vehicle:

```python
-2 $
```

Example:

```text
18 rentals
=> +180 $

3 cars moved
=> -6 $

Reward = 174 $
```

---

# Gym-Style Environment

```python
class CarRentalEnv:

    def __init__(self):
        self.max_cars = 20
        self.max_move = 5

    def step(self, state, action):

        cars_a, cars_b = state

        # Move cars between locations
        cars_a -= action
        cars_b += action

        # Generate rental requests
        req_a = np.random.poisson(3)
        req_b = np.random.poisson(4)

        rent_a = min(cars_a, req_a)
        rent_b = min(cars_b, req_b)

        reward = 10 * (rent_a + rent_b)
        reward -= 2 * abs(action)

        cars_a -= rent_a
        cars_b -= rent_b

        # Generate returns
        cars_a += np.random.poisson(3)
        cars_b += np.random.poisson(2)

        cars_a = min(cars_a, self.max_cars)
        cars_b = min(cars_b, self.max_cars)

        next_state = (cars_a, cars_b)

        return next_state, reward
```