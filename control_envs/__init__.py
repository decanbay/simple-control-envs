from gymnasium.envs.registration import register

register(
    id='Pendulum-v2',
    entry_point='control_envs.envs:PendulumEnv2',
    max_episode_steps=500,
)

register(
    id='PendulumCont-v1',
    entry_point='control_envs.envs:PendulumEnv',
    max_episode_steps=500,
)

register(
    id='CartPole-v2',
    entry_point='control_envs.envs:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPoleContA-v1',
    entry_point='control_envs.envs:CartPoleEnvCont',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPoleSwingUpCont-v1',
    entry_point='control_envs.envs:CartPoleSwingupCont',
    max_episode_steps=1000,  # Longer episodes for swing-up
)

register(
    id='CartPoleSwingUpDisc-v1', 
    entry_point='control_envs.envs:CartPoleSwingupDiscrete',
    max_episode_steps=1000,
)

register(
    id='MSD-v1',
    entry_point='control_envs.envs:MassEnv',
    max_episode_steps=500,
)