from dataclasses import dataclass
import numpy as np
import torch
# import torchviz
import gym
from gym import wrappers
from model import Networks
import collections
import random

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_history = 30000
min_history = 300
tau = 0.02
gamma = 0.99
batch_size = 32
update_period = 4
start_episode = 20


@dataclass
class Step:  # State, Action, Reward, State -> sars
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class history:
    def __init__(self, max_len):
        self.queue = collections.deque(maxlen=max_len)

    def push(self, step):
        self.queue.append(step)

    def sample(self, sample_size):
        ans = random.sample(self.queue, sample_size)
        return ans

    def len(self):
        return len(self.queue)

    def mean(self):
        return float(sum(self.queue)) / float(len(self.queue))


class Agent:
    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        self.env = wrappers.Monitor(self.env, "/home/emile/Videos/", video_callable=(lambda ep: ep % 15 == 0))

        self.networks = Networks(gamma, tau)
        self.history = history(max_history)
        self.global_score = history(50)

        self.global_step = 0
        self.total_play_times = 0

    def play_step(self, state, rand=False):
        if rand:
            act = np.random.uniform(-2.0, 2.0, 1)
        else:
            act = self.networks.select_action(state, noise=True)

        n_state, rew, done, _ = self.env.step(act)
        step = Step(state, act, rew, n_state, done)
        self.history.push(step)
        self.global_step += 1

        return step

    def play(self):
        score = 0.0
        state = self.env.reset()

        while True:
            step = self.play_step(state, (self.total_play_times < start_episode))
            state = step.next_state
            score += step.reward

            if (self.global_step % update_period) == 0:
                if self.history.len() >= min_history:
                    steps = self.history.sample(batch_size)
                    self.networks.train(steps)
                    self.global_step = 0

            if step.done:
                break

        self.total_play_times += 1
        self.global_score.push(score)
        print("train time is {}. ave is {}.".format(self.total_play_times, self.global_score.mean()))

    def train(self):
        while True:
            self.play()


def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()






