import gym
import torch
import cv2
import numpy as np
import itertools

from PIL import Image
from collections import deque


class Memory:
    def __init__(self, environment, memory_size, num_stack, frame_shape):
        self.buffer = {
            'td_target': torch.FloatTensor(np.zeros([memory_size, 1])),
            'advantage': [[0] for _ in range(memory_size)]
        }
        buffer_keys = ['obs', 'action', 'reward', 'terminal', 'action_prob']
        for key in buffer_keys:
            self.buffer[key] = [0 for _ in range(memory_size)]
        self.step_count = 0
        self.pointer = 0
        self.memory_size = memory_size

        self.env = gym.make(environment)
        self.num_stack = num_stack
        self.frame_shape = frame_shape
        self.observation = []

    def __len__(self):
        return len(self.buffer['action'])

    def align_length(self):
        for key, value in self.buffer.items():
            if not isinstance(value, deque):
                self.buffer[key] = value[-self.__len__():]

    def add_memory(self, obs, a, r, t, prob):
        self.buffer['action'][self.pointer] = [a]
        self.buffer['reward'][self.pointer] = [r]
        self.buffer['terminal'][self.pointer] = [1 - t]
        self.buffer['action_prob'][self.pointer] = [prob]
        self.step_count += 1
        self.pointer = (self.pointer + 1) % self.memory_size
        self.buffer['obs'][self.pointer] = obs

    def step(self, pi):
        action_prob = pi(torch.FloatTensor(self.get_state()))
        action = torch.distributions.Categorical(action_prob).sample().item()
        new_obs, reward, done, info = self.env.step(action)
        self.add_memory(new_obs, action, reward/10.0, done, action_prob.flatten()[action].item())
        reward = -1 if done else reward
        # if done:
            # print('terminal!')
        return reward, done

    def reset(self):
        obs = self.env.reset()
        self.buffer['obs'][self.pointer] = obs

    def get_state(self, start_idx=None, end_idx=None, next_state=False):
        obs_list = self.buffer['obs']

        # Return specified slice of states. If not specified, return the last state.
        start_idx = start_idx if start_idx is not None else self.pointer
        end_idx = end_idx if end_idx is not None else self.pointer + 1

        terminal_list = np.array(self.buffer['terminal'])
        terminal_idx = terminal_list.reshape(terminal_list.shape[0])
        states = []
        # print(f'start_idx {start_idx}, end_idx {end_idx}')
        for i in range(start_idx, end_idx):
            # Check whether the past (num_stack-1) obs involve terminal state.
            past_obs_status = [terminal_idx[m] for m in range(i-self.num_stack+1, i)]
            # print(past_obs_status)
            # print(1 in past_obs_status)
            # print(i)
            first_frame_idx = i - self.num_stack + 1 + past_obs_status.index(1) if 1 in past_obs_status else \
                i - self.num_stack + 1
            # print('step count', self.step_count)
            # print(first_frame_idx)
            repeat_obs = obs_list[first_frame_idx]
            num_repeat = past_obs_status.count(0)
            frames = repeat_obs * num_repeat + [obs_list[x] for x in range(first_frame_idx, i+1)]
            states.append(self.process_state(frames))
        states = states[0] if len(states) == 1 else states
        states = states[1:] + [np.zeros(np.array(states[-1]).shape).tolist()] if next_state else states
        # print(states)
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
            return np.asarray(state).reshape(self.num_stack, self.frame_shape[0], self.frame_shape[1])
        else:
            return frames[0]


