import gym
import torch
import random
import numpy as np

MAX_EPISODE_LEN = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

action_selector = {
    'argmax' : lambda action : torch.argmax(action, dim=0).item()
}

class Env:
    def __init__(self, env_id, num_env = 1, action_select = 'argmax', reward_scale = 1e-2):
        self.num_envs = num_env
        self.env = gym.make(env_id)
        try:
            self.state_dim = self.env.observation_space.shape[0]
        except:
            self.state_dim = self.env.observation_space.n
        try:
            self.action_dim = self.env.action_space.n
        except:
            self.action_dim = len(range(int(self.env.action_space.low), int(self.env.action_space.high)))
        
        self.state_std = 1.0
        self.state_mean = 0.0
        self.selector = action_selector[action_select]
        self.rewards_scale = reward_scale
        self.use_mean = False
        self.action_range = [
            torch.tensor(0),
            torch.tensor(self.env.action_space.n),
        ]
        f_states, info = self.env.reset()
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states)
        self._init_return()

    def step(self, action_dist, epoch):
        action = self.selector(action_dist)
        if self.use_mean:
            action = action_dist.mean.reshape(self.num_envs, -1, self.act_dim)[:, -1]
        #action = action.clamp(*self.action_range)
        states, rewards, done, _, info = self.env.step(action)
        states = (
            torch.from_numpy(states).to(device=device).reshape(self.num_envs, -1, self.state_dim)
        )
        self._process(epoch)
        self.states = torch.cat([self.states, states], dim=1)
        self.rewards[:, - 1] = torch.tensor(rewards).to(device=device).reshape(self.num_envs, 1)
        self.action[:, -1] = action
        pred_return = self.rtg[:, -1]  + (rewards * self.rewards_scale)
        self.rtg = torch.cat(
            [self.rtg, pred_return.reshape(self.num_envs, -1, 1)], dim=1
        )
        return self.states, self.action, self.rewards, self.rtg, self.timesteps, done
        
    def _reset_env(self):
        f_states, info = self.env.reset()
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states)
        return f_states
    
    def _init_return(self):
        self._process()

    def _process(self, timestep=None):
        self.action = torch.cat(
            [
                self.action,
                torch.zeros((self.num_envs, self.action_dim), device=device).reshape(
                    self.num_envs, -1, self.action_dim
                ),
            ],
            dim=1,
        )
        self.rewards = torch.cat(
            [
                self.rewards,
                torch.zeros((self.num_envs, 1), device=device).reshape(self.num_envs, -1, 1),
            ],
            dim=1,
        )
        if timestep != None:
            self.timesteps = torch.cat(
                [
                    self.timesteps,
                    torch.ones((self.num_envs, 1), device=device, dtype=torch.long).reshape(
                        self.num_envs, 1
                    )
                    * (timestep + 1),
                ],
                dim=1,
            )



    def _init_output(self, state):
        states = (
        torch.from_numpy(state)
        .reshape(self.num_envs, self.state_dim)
        .to(device=device, dtype=torch.float32)
        ).reshape(self.num_envs, -1, self.state_dim)
        
        actions = torch.zeros(0, device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timesteps = torch.tensor([0] * self.num_envs, device=device, dtype=torch.long).reshape(
            self.num_envs, -1
        )
        target_return = torch.tensor([1]*self.num_envs, dtype=torch.float32).to(device)
        ep = target_return
        target_return = torch.tensor(ep, device=device, dtype=torch.float32).reshape(
            self.num_envs, -1, 1
        )
        return states, actions, rewards, timesteps, target_return


    def reset(self):
        state = self._reset_env()
        self.states = (
        torch.from_numpy(state)
        .reshape(self.num_envs, self.state_dim)
        .to(device=device, dtype=torch.float32)
        ).reshape(self.num_envs, -1, self.state_dim)
        #self.states = torch.cat([self.states, state], dim=1)
        self._init_return()
        return self.states, self.action, self.rtg.float(), self.timesteps
    



