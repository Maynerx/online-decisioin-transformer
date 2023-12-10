from decision_transformer import *
import json
#torch.autograd.set_detect_anomaly(True)


def check_nan(tensor):
    if torch.isnan(tensor).any():
        raise ValueError(f'NaN values encountered')



class DT_PPO:
    def __init__(self, state_dim, action_dim, hidden_size, lr = 3e-4, gamma = 0.99, clip = 0.2, epoch = 10):
        self.dt = DecisionTransformer(state_dim, action_dim, hidden_size)
        self.optimizer = torch.optim.Adam(self.dt.parameters(), lr=lr)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.epoch = epoch
        self.eps_clip = clip
        self.rollout_buffer = RolloutBuffer(buffer_size=50000, state_dim=state_dim, action_dim=action_dim)
    
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

        check_nan(state)
        check_nan(old_action_prob)
        check_nan(reward)
        check_nan(rtg)
        check_nan(timestep)
        

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
        check_nan(next_value)
        check_nan(value)
        adventages = rtg + self.gamma * (1-done) * next_value - value
        
        ratio = (action_preds - old_action_prob).mean()
        surr1 = ratio * adventages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1+ self.eps_clip) * adventages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(value, rtg + self.gamma * (1-done) * next_value.detach())
        entropy = -torch.mean(-action_preds)#-torch.sum(action_preds * (action_preds + 1e-8), dim=1).mean()
        loss = actor_loss + 0.01 * entropy + 0.5 *  critic_loss#actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_(self):
        l = []
        batch_size = 4
        for _ in range(self.epoch):
            batch = self.rollout_buffer.get_batchs(batch_size)
            state = batch['states'].reshape((batch_size, 1, self.state_dim))
            old_action_prob = batch['actions'].reshape((batch_size, 1, self.action_dim))
            reward = batch['rewards'].reshape((batch_size, 1, 1))
            next_state = batch['next_states'].reshape((batch_size, 1, self.state_dim))
            done = batch['dones'].reshape((batch_size, 1)).int()
            rtg = batch['rtg'].reshape((batch_size, 1, 1))
            timestep = batch['timestep'].reshape((batch_size, 1))
            action = batch['great_action'].reshape((batch_size, 1))
    
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
            adventages = reward + self.gamma * (1-done) * next_value - value
            ratio = (action_preds - old_action_prob).mean()
            surr1 = ratio * adventages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1+ self.eps_clip) * adventages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value, reward + self.gamma * (1-done) * next_value.detach())
            entropy = -torch.mean(-action_preds)#-torch.sum(action_preds * (action_preds + 1e-8), dim=1).mean()
            loss = actor_loss + 0.01 * entropy + 0.5 * critic_loss  

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            l.append(loss.item())

        return np.mean(l)
    
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
    
    def Learn(self, timesteps, env, notebook = False, reward_scale = 1e-4, max_timestep = 1000, num_env = 1):
        env = Mult_Env(env, reward_scale=reward_scale, reward_method=BASIC_METHOD, num_env=num_env)
        i = 0
        j = 0
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
                _ ,action_dist = self.get_action(
                state=state,
                action=action,
                rtg=rtg,
                timestep=timestep
                )
                state, action, reward, rtg, timestep, done, great_action = env.step_(action_dist, _)
                self.rollout_buffer.add_experience(old_state, action, reward, state, done, rtg, timestep, great_action)
                loss = self.update()
                losses.append(loss)
                rewards.append(reward.mean().item())
                i += num_env
                k = i/timesteps
                if k >= j:
                    j += 0.1
                    print(f'timestep : {i}, reward_mean_sum : {np.mean(r[5:])}, loss : {loss}')
                f.update(num_env)
                if 1 in done:
                    env.reset()
                    break
                
            r.append(np.sum(rewards))
            r_.append(np.mean(r))
            r__.append(np.mean(r[5:]))
            l.append(np.mean(losses))

        return r, l, r_, r__



import gym

ENV = 'CartPole-v0'
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

LEN_EP = int(7e3)

env.close()

if __name__ == '__main__':

    agent = DT_PPO(state_dim=state_dim, action_dim=action_dim, hidden_size=48, clip=0.3)
    r, l, r_, r__ = agent.Learn(LEN_EP, ENV, notebook=False,reward_scale=1e-2, num_env=2)

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

from stable_baselines3 import PPO