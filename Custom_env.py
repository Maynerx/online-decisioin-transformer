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
        self.use_mean = False
        f_states, info = self.env.reset()
        self.states, self.action, self.rewards, self.timesteps, self.rtg = self._init_output(f_states) 

    def step(self, action_dist, action_range, epoch):
        action = action_dist.sample().reshape(self.num_envs, -1, self.action_dim)[:, -1]
        if self.use_mean:
            action = action_dist.mean.reshape(self.num_envs, -1, self.act_dim)[:, -1]
        action = action.clamp(*action_range)
        states, rewards, done, _, info = self.env.step(action)
        self._process(epoch)
        self.states = torch.cat([self.states, states])
        self.rewards[:, - 1] = torch.from_numpy(rewards).to(device=device).reshape(self.num_envs, 1)
        self.action[:, -1] = action
        pred_return = self.rtg[:, -1]
        self.rtg = torch.cat(
            [self.rtg, pred_return.reshape(self.num_envs, -1, 1)], dim=1
        )
        return self.states, self.action, self.rewards, self.rtg, self.timesteps
        

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
        target_return = ...
        return states, actions, rewards, timesteps, target_return


    def reset(self):
        state, _ = self.env.reset()
        state = (
            torch.from_numpy(state).to(device=device).reshape(self.num_envs, -1, self.state_dim)
        )
        self.states = torch.cat([self.states, state], dim=1)
        return self.states
    




def vec_evaluate_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    use_mean=False,
):
    assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    # ! 1
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

        # the return action is a SquashNormal distribution
        action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
        if use_mean:
            action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )