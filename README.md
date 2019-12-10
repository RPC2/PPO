# PyTorch Implementation of Proximal Policy Optimization (PPO)

### Result

<img src="./asset/score_lr0.002.png" alt="result" style="zoom:80%;" />



OpenAI [defines](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) CartPole as solved "when the average reward is greater than or equal to 195.0 over 100 consecutive trials."

### Hyperparameter used

gamma = 0.99

lambda = 0.95

horizon = 32

k_epoch = 3

learning_rate = 0.002

eps_clip = 0.1

v_coef = 1

entropy_coef = 0.01

### References

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[seungeunrho](https://github.com/seungeunrho)/[minimalRL](https://github.com/seungeunrho/minimalRL)