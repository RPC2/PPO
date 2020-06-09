import gym
import torch
import cv2
import numpy as np

from PIL import Image
from collections import deque


class Memory:
    def __init__(self, environment, memory_size, num_stack, frame_shape):
        self.buffer = {
            'td_target': torch.FloatTensor([])
        }
        buffer_keys = ['obs', 'action', 'reward', 'terminal', 'action_prob']
        for key in buffer_keys:
            self.buffer[key] = deque(maxlen=memory_size)

        self.env = gym.make(environment)
        self.num_stack = num_stack
        self.frame_shape = frame_shape

    def add_memory(self, s, a, r, t, prob):
        self.buffer['obs'].append(s)
        self.buffer['action'].append([a])
        self.buffer['reward'].append([r])
        self.buffer['terminal'].append([1 - t])
        self.buffer['action_prob'].append(prob)

    def step(self, action_prob):
        action = torch.distributions.Categorical(action_prob).sample().item()
        observation, reward, done, info = self.env.step(action)
        reward = -1 if done else reward
        self.add_memory(observation, action, reward/10.0, done, action_prob[action].item())
        return reward, done

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.num_stack):
            self.buffer['obs'].append(observation)
        for i in self.buffer:
            if isinstance(i, deque):
                i.clear()

    def get_state(self, return_all=False):
        start_idx = 3 if return_all else len(self.buffer['obs'])-1
        states = []
        for i in range(start_idx, len(self.buffer['obs'])):
            frames = self.buffer['obs'][i-(self.num_stack-1):i+1]
            states.append(self.process_state(frames))
        return states

    def process_state(self, frames):
        """Returns a stack of frames with shape [num_stack, width, height]"""
        if len(self.frame_shape) == 2:
            state = []
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = cv2.resize(frame, self.frame_shape, interpolation=cv2.INTER_AREA)
                frame = np.expand_dims(frame, -1)
                state.append(frame)
                # plt.imshow(frame)
                # plt.show()
        else:
            state = frames
        return np.asarray(state).reshape(self.num_stack, self.frame_shape[0], self.frame_shape[1])


