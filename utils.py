import torch
import matplotlib.pyplot as plt


def plot_graph(reward_history):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')

    plt.plot(range(len(reward_history)), reward_history, marker='', color=palette(1), linewidth=0.8, alpha=0.9,
             label='Reward')

    # plt.legend(loc='upper left')
    plt.title("CartPole", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    # plt.ylabel("score", fontsize=12)

    plt.savefig('score.png')


def save_weights(model):
    torch.save(model.state_dict(), 'expert_cp.ckpt')