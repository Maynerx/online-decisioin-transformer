import gym
import torch
import random
import numpy as np

MAX_EPISODE_LEN = 1000

def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret



class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories

    def __getitem__(self, index):
        traj = self.trajs[self.sampling_ind[index]]
        return traj

    def __len__(self):
        return len(self.sampling_ind)

def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    num_workers=24,
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size)

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind)

    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

def sample_trajs(trajectories, sample_size):
    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)
    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Env:
    def __init__(self, env_id, num_env = 1):
        self.num_envs = num_env
        self.env = gym.make(env_id)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.state_std = 1.0
        self.state_mean = 0.0
        self.rewards_scale = 1
        self.use_mean = False
        self.action_range = [
            float(self.env.action_space.low.min()) + 1e-6,
            float(self.env.action_space.high.max()) - 1e-6,
        ]
        f_states, info = self.env.reset()
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states) 

    def step(self, action_dist, epoch):
        action = action_dist.sample().reshape(self.num_envs, -1, self.action_dim)[:, -1]
        if self.use_mean:
            action = action_dist.mean.reshape(self.num_envs, -1, self.act_dim)[:, -1]
        action = action.clamp(*self.action_range)
        states, rewards, done, _, info = self.env.step(action)
        self._process(epoch)
        self.states = torch.cat([self.states, states])
        self.rewards[:, - 1] = torch.from_numpy(rewards).to(device=device).reshape(self.num_envs, 1)
        self.action[:, -1] = action
        pred_return = self.rtg[:, -1]  - (rewards * self.rewards_scale)
        self.rtg = torch.cat(
            [self.rtg, pred_return.reshape(self.num_envs, -1, 1)], dim=1
        )
        return self.states, self.action, self.rewards, self.rtg, self.timesteps, done
        
    def _reset_env(self):
        f_states, info = self.env.reset()
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states)
        return f_states

    def _process(self, timestep):
        self.actions = torch.cat(
            [
                self.actions,
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
        target_return = torch.tensor([1]*self.num_envs)
        return states, actions, rewards, timesteps, target_return


    def reset(self):
        state = self._reset_env()
        state = (
            torch.from_numpy(state).to(device=device).reshape(self.num_envs, -1, self.state_dim)
        )
        self.states = torch.cat([self.states, state], dim=1)
        return self.states
    



