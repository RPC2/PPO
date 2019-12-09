class AgentConfig:
    # Learning
    gamma = 0.99
    # train_freq = 1
    # start_learning = 10
    # batch_size = 32
    plot_every = 100
    # reset_step = 100
    horizon = 32

    # epsilon = 0
    # epsilon_minimum = 0
    # epsilon_decay_rate = 0.9995
    k_epoch = 3
    learning_rate = 0.002
    lmbda = 0.95
    eps_clip = 0.1
    v_coef = 1
    entropy_coef = 0.01

    max_step = 40000000       # 40M steps max
    max_episode_length = 18000  # equivalent of 5 minutes of game play at 60 frames per second

    # Algorithm selection
    train_cartpole = True
    per = False

    double_q_learning = False
    duelling_dqn = False

    gif = False
    gif_every = 9999999


class EnvConfig:
    env_name = 'CartPole-v0'
    save_every = 10000
