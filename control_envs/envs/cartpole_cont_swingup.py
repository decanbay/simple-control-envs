import math
import gymnasium as gym  # Changed from gym
from gymnasium import spaces, logger  # Changed from gym
from gymnasium.utils import seeding  # Changed from gym.utils
import numpy as np


class CartPoleSwingupCont(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, mc=1.0, mp=0.1, l=0.5, max_force=10.0, integrator='semi-implicit', random_start=True):
        super().__init__()
        self.gravity = 9.8
        self.masscart = mc
        self.masspole = mp
        self.total_mass = mp + mc
        self.length = l  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = max_force
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = integrator
        self.random_start = random_start
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        x, x_dot, theta, theta_dot = self.state
        #force = self.force_mag if action == 1 else -self.force_mag
        force = action[0]
        force = np.clip(force, -self.force_mag, self.force_mag)
        theta = self._angle_normalize(theta) ###

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        theta = self._angle_normalize(theta) ### 

        self.state = (x, x_dot, theta, theta_dot)

        # Swing-up reward: reward for being upright and centered
        # Normalize theta to [-pi, pi]
        theta_normalized = self._angle_normalize(theta)
        
        # Reward for being upright (theta close to 0) and centered (x close to 0)
        upright_reward = np.cos(theta_normalized)  # 1 when upright, -1 when hanging
        centered_reward = np.exp(-x**2)  # 1 when centered, decreases with distance
        
        reward = upright_reward + 0.1 * centered_reward - 0.001 * force**2

        # Episode ends if cart goes too far
        terminated = bool(x < -self.x_threshold or x > self.x_threshold)

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start with pole hanging down and cart slightly off-center
        if options and 'random_start' in options:
            self.state = self.np_random.uniform(low=[-0.1, -0.1, np.pi-0.1, -0.1], 
                                              high=[0.1, 0.1, np.pi+0.1, 0.1])
        else:
            # Default: pole hanging down
            self.state = np.array([0.0, 0.0, np.pi, 0.0])
            
        return np.array(self.state, dtype=np.float32), {}

    def _angle_normalize(self, x):
        """Normalize angle to [-pi, pi]"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def render(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleSwingupDiscrete(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, mc=1.0, mp=0.1, l=0.5, max_force=10.0, integrator='semi-implicit', random_start=True):
        super(gym.Env).__init__()
        self.gravity = 9.8
        self.masscart = mc
        self.masspole = mp
        self.total_mass = mp + mc
        self.length = l  # actually half the pole's length
        self.polemass_length = mp * l
        self.force_mag = max_force
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = integrator
        self.random_start = random_start
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
    
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
    
        # 5 discrete actions: large left, small left, no force, small right, large right
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
    
        self.seed()
        self.viewer = None
        self.state = None
    
        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def step(self, action):
        # Map discrete actions to continuous forces
        force_map = {
            0: -self.force_mag,      # Large left
            1: -self.force_mag/2,    # Small left  
            2: 0.0,                  # No force
            3: self.force_mag/2,     # Small right
            4: self.force_mag        # Large right
        }
        
        force = force_map[action]
        
        x, x_dot, theta, theta_dot = self.state
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # Swing-up reward
        theta_normalized = self._angle_normalize(theta)
        upright_reward = np.cos(theta_normalized)
        centered_reward = np.exp(-x**2)
        
        reward = upright_reward + 0.1 * centered_reward - 0.001 * force**2

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'random_start' in options:
            self.state = self.np_random.uniform(low=[-0.1, -0.1, np.pi-0.1, -0.1], 
                                              high=[0.1, 0.1, np.pi+0.1, 0.1])
        else:
            self.state = np.array([0.0, 0.0, np.pi, 0.0])
            
        return np.array(self.state, dtype=np.float32), {}

    def _angle_normalize(self, x):
        """Normalize angle to [-pi, pi]"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def render(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None