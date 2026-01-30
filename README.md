# Simple Control Environments

A collection of simple control environments for reinforcement learning (RL) tasks, compatible with Gymnasium (formerly OpenAI Gym).

## Description

This package provides custom Gymnasium environments for classic control problems, including both continuous and discrete action spaces. These environments are designed for testing and developing reinforcement learning algorithms.

## Features

- **Multiple Control Environments**: Pendulum, CartPole, and Mass-Spring-Damper systems
- **Continuous and Discrete Actions**: Various action space configurations
- **Gymnasium Compatible**: Works with the latest Gymnasium API
- **Customizable Parameters**: Adjustable physical parameters for each environment
- **Swing-up Variants**: Challenging swing-up tasks for CartPole

## Installation

### From Source

Clone the repository and install:

```bash
git clone https://github.com/decanbay/simple-control-envs.git
cd simple-control-envs
pip install -e .
```

### Requirements

- Python >= 3.7
- gymnasium
- numpy
- matplotlib

## Available Environments

### Pendulum Environments

- **`Pendulum-v2`**: Classic pendulum environment with discrete actions
- **`PendulumCont-v1`**: Pendulum with continuous action space

### CartPole Environments

- **`CartPole-v2`**: Classic CartPole with discrete actions
- **`CartPoleContA-v1`**: CartPole with continuous action space
- **`CartPoleSwingUpCont-v1`**: Continuous action swing-up task (1000 steps)
- **`CartPoleSwingUpDisc-v1`**: Discrete action swing-up task (1000 steps)

### Mass-Spring-Damper

- **`MSD-v1`**: Mass-Spring-Damper system with continuous actions

## Usage

### Basic Example

```python
import gymnasium as gym
import control_envs

# Create environment
env = gym.make('PendulumCont-v1')

# Reset environment
observation, info = env.reset()

# Run episode
for _ in range(500):
    # Random action
    action = env.action_space.sample()
    
    # Step environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render (optional)
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Custom Parameters

Environments support custom physical parameters:

```python
import gymnasium as gym
import control_envs

# Create pendulum with custom parameters
env = gym.make('PendulumCont-v1', m=1.5, l=0.8, max_torque=3.0)
```

### CartPole Swing-up

The swing-up variants are more challenging tasks where the pole starts hanging down:

```python
import gymnasium as gym
import control_envs

# Continuous swing-up task
env = gym.make('CartPoleSwingUpCont-v1')

# Discrete swing-up task
env = gym.make('CartPoleSwingUpDisc-v1')
```

## Environment Details

### Action Spaces

- **Continuous**: Box action space with specified ranges
- **Discrete**: Discrete action space with limited actions

### Observation Spaces

Varies by environment, typically including:
- Position and velocity
- Angle and angular velocity (for pendulum/pole)
- Trigonometric representations of angles

### Episode Length

- Standard environments: 500 steps
- Swing-up environments: 1000 steps

## Development

### Project Structure

```
simple-control-envs/
├── control_envs/          # Main package
│   ├── __init__.py        # Environment registration
│   └── envs/              # Environment implementations
│       ├── pendulum.py
│       ├── pendulum_cont.py
│       ├── cartpole.py
│       ├── cartpole_cont.py
│       ├── cartpole_cont_swingup.py
│       └── MassSpringDamper.py
├── setup.py               # Package setup
└── README.md              # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Author

**Deniz Ekin Canbay**
- Email: decanbay@gmail.com

## Acknowledgments

- Built on top of [Gymnasium](https://gymnasium.farama.org/)
- Inspired by classic control theory problems
