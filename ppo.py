from decision_transformer import *
import json
#torch.autograd.set_detect_anomaly(True)
#torch.manual_seed(2023)

def check_nan(tensor):
    if torch.isnan(tensor).any():
        raise ValueError(f'NaN values encountered')

def schedule(s : torch.optim.lr_scheduler.StepLR, step, max_step):
    if (step / max_step) < 0.8:
        return s.step()

class DT_PPO:
    def __init__(self, state_dim, action_dim, hidden_size, lr = 3e-4, gamma = 0.99, clip = 0.2, epoch = 10, buffer_size = 50000, nlayer = 3, nhead = 3, max_norm = 0.2, normalize = True):
        self.dt = DecisionTransformer(state_dim, action_dim, hidden_size, nlayer=nlayer, nhead=nhead)
        self.dt2 = DecisionTransformer(state_dim, action_dim, hidden_size, nlayer=nlayer, nhead=nhead)
        self.dt2.load_state_dict(self.dt.state_dict())
        self.optimizer = torch.optim.Adam(self.dt.parameters(), lr=lr)
        self.max_norm = max_norm
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.normalize = normalize
        self.epoch = epoch
        self.eps_clip = clip
        self.rollout_buffer = Buffer()#RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim)
    
    def get_action(self, state, action,rtg, timestep):
        action, action_logprob, action_preds, value = self.dt.get_action(
            state, 
            action,
            None,
            rtg,
            timesteps=timestep
            )
        return action, action_logprob, action_preds, value

        
    def update(self):
        pass
    
    
    def Learn_(self, timesteps, env, notebook = False, reward_scale = 1e-4, max_timestep = 200):
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=timesteps//10, gamma=0.1)
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
                action, action_log_prob, action_pred, old_value = self.get_action(
                state=state,
                action=action,
                rtg=rtg,
                timestep=timestep
                )
                state, action, reward, rtg, timestep, done, great_action = env.step_(action, _)
                self.rollout_buffer.add_experience_(state, action_pred, reward, state, done, rtg, timestep, great_action, action_log_prob, old_value)
                loss = self.update()
                #schedule(self.schedule, i, timesteps)
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
    
 
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_size = 64, nlayer = 1, nhead = 1):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = Buffer()
        self.policy = ActorCritic(state_dim, action_dim, hidden_size, nlayer=nlayer, nhead=nhead)
        self.optimizer = torch.optim.Adam([
            {'params' : self.policy.actor.parameters(), 'lr' : lr_actor},
            {'params' : self.policy.critic.parameters(), 'lr' : lr_critic}           
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_size, nlayer=nlayer, nhead=nhead)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.criterion = nn.MSELoss()

    def select_action(self, state, action, returns_to_go, timesteps):
        with torch.no_grad():
            state = torch.tensor(state).float()
            action, action_logprob, state_val, action_pred = self.policy.select_actions(state, action, returns_to_go, timesteps)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.action_pred.append(action_pred)
        return action.item()
    
    def update(self):
        # Monte Carlo estimate of returns
        losses = []
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(DEVICE)
        old_rtg = torch.squeeze(torch.stack(self.buffer.rtg, dim=0)).detach().to(DEVICE)
        old_timesteps = torch.squeeze(torch.stack(self.buffer.timesteps, dim=0)).detach().to(DEVICE)
        old_actions_preds = torch.squeeze(torch.stack(self.buffer.action_pred, dim=0)).detach().to(DEVICE)


        old_states = old_states.reshape((self.buffer.__len__(), 1, old_states.size(1)))
        old_rtg = old_rtg.reshape((self.buffer.__len__(), 1, 1))
        old_timesteps = old_timesteps.reshape((self.buffer.__len__(), 1))
        old_actions_preds = old_actions_preds.reshape((self.buffer.__len__(), 1, old_actions_preds.size(1)))

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_actions, old_rtg, old_timesteps, old_actions_preds)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.criterion(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            losses.append(loss.mean().item())
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return np.mean(losses)


def n_p(model):
    return sum(p.numel() for p in model.policy.parameters()) / 1e3

#print(n_p(PPO(1, 1, 1e-1, 1e-1, 0.99, 80, 0.2, hidden_size=10, nlayer=1, nhead=1)))

'''
import gym

ENV = 'CartPole-v1'
env = gym.make(ENV)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

LEN_EP = int(1.5e4)

env.close()


agent = DT_PPO(state_dim=state_dim, action_dim=action_dim, hidden_size=64, clip=0.2, epoch=10, gamma=0.99, buffer_size=100_000, lr=3e-4, nhead = 1, nlayer = 1, normalize = True, max_norm = 1)
r, l, r_, r__ = agent.Learn_(LEN_EP, ENV, notebook=False ,reward_scale=-0.5, max_timestep=5000)

L = len(r)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Absolute reward
axs[0, 0].plot(range(L), r, label='Absolute reward', color='blue')
axs[0, 0].set_title('Absolute Reward')
axs[0, 0].set(xlabel='Number of Episodes', ylabel='Reward')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Loss evolution
axs[0, 1].plot(range(L), l, label='Loss Evolution', color='orange')
axs[0, 1].set_title('Loss Evolution')
axs[0, 1].set(xlabel='Number of Episodes', ylabel='Loss')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Average reward
axs[1, 0].plot(range(L), r_, label='Average Reward', color='green')
axs[1, 0].set_title('Average Reward')
axs[1, 0].set(xlabel='Number of Episodes', ylabel='Reward')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Average reward (cut 5)
axs[1, 1].plot(range(L), r__, label='Average Reward Cut 5', color='red')
axs[1, 1].set_title('Average Reward Cut 5')
axs[1, 1].set(xlabel='Number of Episodes', ylabel='Reward')
axs[1, 1].grid(True)
axs[1, 1].legend()

# Common title
plt.suptitle('Agent Learning Performance', fontsize=16)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()
'''