from decision_transformer import *
import json
#torch.autograd.set_detect_anomaly(True)


def check_nan(tensor):
    if torch.isnan(tensor).any():
        raise ValueError(f'NaN values encountered')



class DT_PPO:
    def __init__(self, state_dim, action_dim, hidden_size, lr = 3e-4, gamma = 0.99, clip = 0.2, epoch = 10, buffer_size = 50000):
        self.dt = DecisionTransformer(state_dim, action_dim, hidden_size)
        self.optimizer = torch.optim.Adam(self.dt.parameters(), lr=lr)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.epoch = epoch
        self.eps_clip = clip
        self.rollout_buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim)
    
    def get_action(self, state, action,rtg, timestep):
        action_dist = self.dt.get_action(
            state, 
            action,
            None,
            rtg,
            timesteps=timestep
            )
        return action_dist
    
    def update(self):
        batch = self.rollout_buffer.get_batch()
        state = batch['states']
        old_action_prob = batch['actions']
        reward = batch['rewards']
        next_state = batch['next_states']
        done = batch['dones']
        rtg = batch['rtg']
        timestep = batch['timestep']
        action = batch['great_action']

        _, action_preds, return_preds, value = self.dt.forward(
                state,
                old_action_prob,
                None,
                rtg,
                timestep
            )
        s_, _, _1, next_value = self.dt.forward(
                next_state,
                action_preds,
                None,
                rtg,
                timestep
            )
        advantages = rtg + self.gamma * (1-done) * next_value - value
        ratio = (action_preds - old_action_prob).mean()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1+ self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(value, rtg + self.gamma * (1-done) * next_value.detach())
        entropy = -torch.mean(-action_preds)#-torch.sum(action_preds * (action_preds + 1e-8), dim=1).mean()
        loss = actor_loss + 0.5 * entropy + 0.6 *  critic_loss#actor_loss + 0.5 * critic_loss - 0.01 * entropy            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.dt.parameters(), max_norm=0.5)

        return loss.item()
    
    
    def save(self, path):
        torch.save(self.dt, f'{path}/model.pt')
        with open(f'{path}/param.json', 'w+') as f:
            d = {
                'lr' : self.lr,
                'gamma' : self.gamma,
                'state_dim' : self.state_dim,
                'action_dim' : self.action_dim,
                'eps_clip' : self.eps_clip
            }
            json.dump(d, f)
        

    def load(path):
        dt = torch.load(f'{path}/model.pt')
        with open(f'{path}/param.json', 'r+') as f:
            param = json.load(f)
        lr = param['lr']
        gamma = param['gamma']
        state_dim = param['state_dim']
        action_dim = param['action_dim']
        eps_clip = param['eps_clip']
        ppo = DT_PPO(state_dim, action_dim, 12, lr, gamma, eps_clip)
        ppo.dt = dt
        return ppo
    
    def Learn(self, timesteps, env, notebook = False, reward_scale = 1e-4, max_timestep = 1000):
        env = Env(env, reward_scale=reward_scale, reward_method=BASIC_METHOD)
        i = 0
        r, l, r_, r__ = [], [], [], []
        losses = []
        f = tqdm.tqdm(total=timesteps) if not notebook else tqdm_notebook(total=timesteps)
        while i <= timesteps:
            state, action, rtg, timestep = env.reset()
            rewards = []
            state_std, state_mean = state.std(), state.mean()
            #print(self.optimizer.param_groups)
            for _ in range(max_timestep):
                old_state = state.clone()
                action_dist = self.get_action(
                state=state,
                action=action,
                rtg=rtg,
                timestep=timestep
                )
                state, action, reward, rtg, timestep, done, great_action = env.step_(action_dist, _)
                self.rollout_buffer.add_experience(old_state, action, reward, state, done, rtg, timestep, great_action)
                loss = self.update()
                losses.append(loss)
                rewards.append(reward.squeeze(2)[0][-1].item())
                i += 1
                if i % (timesteps // 10) == 0: 
                    print(f'timestep : {i}, reward_mean_sum : {np.mean(r[-2:])}, loss : {loss}')
                f.update(1)
                if done:
                    env.reset()
                    break
                
            r.append(np.sum(rewards))
            r_.append(np.mean(r))
            r__.append(np.mean(r[-5:]))
            l.append(np.mean(losses))

        return r, l, r_, r__

'''

import gym

ENV = 'CartPole-v1'
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

LEN_EP = int(1e5)

env.close()


agent = DT_PPO(state_dim=state_dim, action_dim=action_dim, hidden_size=24, clip=0.3, epoch=1, gamma=0.9, buffer_size=500_000)
r, l, r_, r__ = agent.Learn(LEN_EP, ENV, notebook=False,reward_scale=1e-1)

L = len(r)

fig, axs = plt.subplots(4)
axs[0].plot(range(L), r)
axs[0].set_title('Absolute reward')
axs[0].set(xlabel = 'num_episodes', ylabel = 'reward')
axs[1].plot(range(L), l, 'tab:orange')
axs[1].set_title('Loss evolution')
axs[1].set(xlabel = 'num_episodes', ylabel = 'loss')
axs[2].plot(range(L), r_, 'tab:green')
axs[2].set_title('Av reward')
axs[2].set(xlabel = 'num_episodes', ylabel = 'reward')
axs[3].set_title('Av reward cut 5')
axs[3].set(xlabel = 'num_episodes', ylabel = 'reward')
axs[3].plot(range(L), r__, 'tab:red')


plt.show()

#agent.save('bin')
'''