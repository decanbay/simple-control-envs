from gym.envs.registration import register

register(
    id='PendulumCont-v1',
    entry_point='control_envs.envs:PendulumEnv',
    max_episode_steps=200,
    nondeterministic=False
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
    max_episode_steps=500
)
register(
    id='CartPoleSwingUpDisc-v1',
    entry_point='control_envs.envs:CartPoleSwingupDiscrete',
    max_episode_steps=500
)
register(
    id='PendulumDisc-v1',
    entry_point='control_envs.envs:PendulumDiscEnv',
    max_episode_steps=200,
    nondeterministic=False
)
register(
    id='MSD-v1',
    entry_point='control_envs.envs:MassEnv',
    max_episode_steps=250,
    nondeterministic=False
)