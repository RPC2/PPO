class AgentConfig:
    # Learning
    gamma = 0.99
    plot_every = 100
    horizon = 32

    k_epoch = 3
    learning_rate = 0.002
    lmbda = 0.95
    eps_clip = 0.1
    v_coef = 1
    entropy_coef = 0.01

    max_step = 40000000       # 40M steps max
    max_episode_length = 18000  # equivalent of 5 minutes of game play at 60 frames per second

    gif = False
    gif_every = 9999999


class EnvConfig:
    env_name = 'CartPole-v0'
    save_every = 10000
