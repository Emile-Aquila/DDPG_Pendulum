import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ANetwork(torch.nn.Module):
    def __init__(self):
        super(ANetwork, self).__init__()
        # define network
        self.d1 = nn.Linear(3, 64)
        self.f1 = nn.ReLU()
        self.d2 = nn.Linear(64, 64)
        self.f2 = nn.ReLU()
        self.d3 = nn.Linear(64, 1)
        self.f3 = nn.Tanh()

        self.train()

    def forward(self, inputs):
        inputs = torch.from_numpy(inputs).float().to(dev)
        x = self.f1(self.d1(inputs))
        x = self.f2(self.d2(x))
        x = self.f3(self.d3(x))
        x = x * 2.0
        return x


class CNetwork(torch.nn.Module):
    def __init__(self):
        super(CNetwork, self).__init__()
        # define network
        self.d1 = nn.Linear(3 + 1, 64)
        self.f1 = nn.ReLU()
        self.d2 = nn.Linear(64, 64)
        self.f2 = nn.ReLU()
        self.d3 = nn.Linear(64, 1)

        self.train()

    def forward(self, state, action):
        # inputs = np.concatenate((state, action), axis=0)
        state = torch.from_numpy(state).float().to(dev)
        # inputs = np.concatenate((state, np.array(action)), axis=0)
        # inputs = torch.from_numpy(inputs).float()
        inputs = torch.cat([state, action], dim=0)
        x = self.f1(self.d1(inputs))
        x = self.f2(self.d2(x))
        x = self.d3(x)
        return x


class Networks:
    def __init__(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau

        # define actor/critic and each target networks.
        self.actor = ANetwork().to(dev)
        self.actor_target = ANetwork().to(dev)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = CNetwork().to(dev)
        self.critic_target = CNetwork().to(dev)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        for param, t_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            if (t_param.data is None) or (param.data is None):
                continue
            t_param.data = param.data.clone()

        for param, t_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            if (t_param.data is None) or (param.data is None):
                continue
            t_param.data = param.data.clone()

    def select_action(self, state, noise=False):
        act = self.actor(state).to(dev).detach().numpy()
        if noise:
            act += np.random.normal(0, 2.0 * 0.2, size=1)
            act = np.clip(act, -2.0, 2.0)
        return act

    def calc_loss_actor(self, steps):
        loss = []
        for step in steps:
            tmp = self.critic(step.state, self.actor(step.state)).to(dev)
            loss.append(tmp)

        loss = torch.stack(loss, dim=0).to(dev)
        loss = -1 * torch.sum(loss, dim=0).to(dev)
        return loss

    def calc_loss_critic(self, steps):
        loss = []
        for step in steps:
            if not step.done:
                tmp = self.critic_target(step.next_state, self.actor_target(step.next_state).detach()).to(dev).detach()
            else:
                tmp = torch.Tensor([0.0])
            tmp = (torch.Tensor([step.reward]) + self.gamma * tmp).to(dev)
            tmp = tmp - self.critic(step.state, torch.from_numpy(step.action).float()).to(dev)
            loss.append(tmp)

        # print("loss {}".format(loss))
        loss = torch.stack(loss, dim=0).to(dev)
        # print("loss 1 {}".format(loss))
        loss = (loss**2).to(dev)
        # print("loss 2 {}".format(loss))
        loss = torch.sum(loss, dim=0).to(dev)
        return loss

    def update_target(self):
        for param, t_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            if (t_param.data is None) or (param.data is None):
                continue
            t_param.data = (1.0 - self.tau) * t_param.data + self.tau * param.data.clone()

        for param, t_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            if (t_param.data is None) or (param.data is None):
                continue
            t_param.data = (1.0 - self.tau) * t_param.data + self.tau * param.data.clone()

    def train(self, steps):
        critic_loss = self.calc_loss_critic(steps)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self.calc_loss_actor(steps)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target()




