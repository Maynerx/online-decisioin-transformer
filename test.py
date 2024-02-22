from ppo import *
from datetime import datetime


def train(env, path, max_training_timesteps = 200_000, render = False, gamma = 0.99, lr_actor = 0.0003, lr_critic = 0.001, notebook = False):
    print("============================================================================================")
    env_name = env
    max_ep_len = 1000 
    print_freq = max_ep_len * 10
    update_timestep = max_ep_len * 4 
    K_epochs = 80
    eps_clip = 0.2
    hidden_size = 10
    print("training environment name : " + env_name)
    rewards = []
    losses = []
    steps = []
    if render:
        pass
    else:
        env = Env(env_name)
    state_dim = env.env.observation_space.shape[0]

    action_dim = env.env.action_space.n
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    print("============================================================================================")
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_size=hidden_size)
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    f = tqdm.tqdm(total=max_training_timesteps) if not notebook else tqdm_notebook(total=max_training_timesteps)
    while time_step <= max_training_timesteps:

        state, action, rtg, timestep = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state, action, rtg, timestep)
            state, action, reward, rtg, timestep, done, great_action = env.step(action)
            if render:
                env.render()
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            ppo_agent.buffer.rtg.append(rtg)
            ppo_agent.buffer.timesteps.append(timestep)

            time_step +=1
            f.update(1)
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                loss = ppo_agent.update()
                losses.append(loss)



            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # break; if the episode is over
            
            if done:
                break
        steps.append(t)
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1
        rewards.append(round(print_running_reward / print_running_episodes, 2))

        i_episode += 1

    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    return rewards, losses, steps


"""
rewards, losses, steps = train('CartPole-v1', '', max_training_timesteps=700_000, lr_actor=3e-4, lr_critic=1e-3)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

L = len(rewards)

# Absolute reward
axs[0, 0].plot(range(L), rewards, label='Absolute reward', color='blue')
axs[0, 0].set_title('Absolute Reward')
axs[0, 0].set(xlabel='Number of Episodes', ylabel='Reward')
axs[0, 0].grid(True)
axs[0, 0].legend()

M = len(losses)
# Loss evolution
axs[0, 1].plot(range(M), losses, label='Loss Evolution', color='orange')
axs[0, 1].set_title('Loss Evolution')
axs[0, 1].set(xlabel='Number of Episodes', ylabel='Loss')
axs[0, 1].grid(True)
axs[0, 1].legend()


# Common title
plt.suptitle('Agent Learning Performance', fontsize=16)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()
"""