import torch
import numpy as np
import gymnasium as gym
import random
import os
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device('cuda')
else:
    print("Using CPU...")
    device = torch.device('cpu')

class ValueNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features = 1) -> None:
        super().__init__()
        self.ff1 = torch.nn.Linear(in_features, 128)
        self.activation_1 = torch.nn.GELU()
        self.ff2 = torch.nn.Linear(128, 128)
        self.activation_2 = torch.nn.GELU()
        self.out = torch.nn.Linear(128, out_features)
    
    def forward(self, state):
        x = self.ff1(state)
        x = self.activation_1(x)
        x = self.ff2(x)
        x = self.activation_2(x)
        value = self.out(x)
        return value

class PolicyNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.ff1 = torch.nn.Linear(in_features, 128)
        self.activation_1 = torch.nn.GELU()
        self.ff2 = torch.nn.Linear(128, 128)
        self.activation_2 = torch.nn.GELU()
        self.out = torch.nn.Linear(128, out_features)
    
    def forward(self, state):
        x = self.ff1(state)
        x = self.activation_1(x)
        x = self.ff2(x)
        x = self.activation_2(x)
        x = self.out(x)
        return x

class PPOAgent():
    def __init__(self, state_shape, action_shape) -> None:
        self.value_network = None
        self.policy_network = None
        self.old_policy_network = None
    
    def update_old_policy_network(self):
        pass

    def sample_action(self):
        pass

class PPOAgent_torch(PPOAgent):
    def __init__(self, state_shape, action_shape) -> None:
        self.value_network = ValueNetwork(state_shape, 1).to(device)
        self.policy_network = PolicyNetwork(state_shape, action_shape).to(device)
        self.old_policy_network = PolicyNetwork(state_shape, action_shape).requires_grad_(False).to(device)
        self.update_old_policy_network()

    def sample_action(self, obs):
        a = self.policy_network(obs)
        a_dist = torch.distributions.Categorical(logits=a)
        return a_dist.sample()
    
    def update_old_policy_network(self):
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())
    
    def save(self, path):
        torch.save({
                    'policy_network': self.policy_network.state_dict(),
                    'value_network': self.value_network.state_dict()
                    }, path)
    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.update_old_policy_network()
        self.value_network.load_state_dict(checkpoint['value_network'])



class ExperienceBuffer():
    def __init__(self) -> None:
        self.buffer = []
        self.shuffled_indices = []
        self.start_index = 0

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int, T_step: int):
        # if batches are exausted, shuffle indices again
        if len(self.shuffled_indices) - self.start_index < T_step + batch_size:
            self.shuffled_indices = list(range(len(self.buffer) - T_step))
            random.shuffle(self.shuffled_indices)
            self.start_index = 0
        # if there are not enough samples, return None immediately
        if len(self.shuffled_indices) - self.start_index < T_step + batch_size:
            return
        
        sample_indices = self.shuffled_indices[self.start_index:self.start_index+batch_size]

        start_obs_batch = []
        end_obs_batch = []
        reward_batch = []
        action_batch = []
        terminated_batch = []

        # n_step bootstrapping
        # 0 = obs, 1 = action, 2 = reward, 3 = terminated
        for i in sample_indices:
            start_obs = self.buffer[i][0]
            action = self.buffer[i][1]
            end_obs = self.buffer[i+T_step][0]
            term_ = False
            terminated = []
            rewards = []
            for index, step in enumerate(self.buffer[i:i+T_step]):
                rewards.append(step[2])
                # if an episode is terminated, set end_obs to the terminal observation
                if step[3] == True:
                    end_obs = step[0]
                    term_ = True
                    for _ in range(index + 1, T_step):
                        rewards.append(0.)
                    break
            terminated.append(term_)

            start_obs_batch.append(np.array(start_obs))
            end_obs_batch.append(np.array(end_obs))
            reward_batch.append(np.array(rewards))
            action_batch.append(np.array(action))
            terminated_batch.append(np.array(terminated).astype(np.float32))
            

        # increase start_index by batch_size
        self.start_index += batch_size
        
        return np.stack(start_obs_batch, axis=0), np.stack(end_obs_batch, axis=0), np.stack(reward_batch, axis=0), np.stack(action_batch), np.stack(terminated_batch)

    def push(self, obs, action, reward, done):
        self.buffer.append((obs, action, reward, done))

    def clear(self):
        self.buffer.clear()
        self.shuffled_indices.clear()
        self.start_index = 0
'''
    Runs training loop for a certain amount. 
    Specifically,
    1. for each loop in num_run times, sample from the buffer, and update policy using backpropagation.
    2. After the loop is done, clear the buffer.
'''
def PPO_trainer(agent: PPOAgent, expBuffer: ExperienceBuffer, optimizer: torch.optim.Optimizer, T_step: int, batch_size: int, num_runs: int = None, clip_epsilon: float = 0.2, discount_gamma: float = 0.99, entropy_coef: float = 0.01):
    # when should a loop end? lets try using this simple method.
    # would it be better to use the probability ratio of (pi / old pi) to decide when to stop the loop?
    if num_runs == None:
        num_runs = len(expBuffer) // batch_size

    # loop start
    for i in range(num_runs):
        start_obs, end_obs, n_reward, action, terminated = map(torch.tensor, expBuffer.sample(batch_size, T_step))

        start_obs = start_obs.float().to(device)
        end_obs = end_obs.float().to(device)
        n_reward = n_reward.float().to(device)
        action = action.float().to(device)
        terminated = terminated.float().to(device)

        with torch.no_grad():
            # generalized advantage estimation, TD(n)
            # action-value estimation
            reward_rollout = torch.concat([n_reward, agent.value_network(end_obs) * (1 - terminated)], dim=-1)

            action_value_est = torch.matmul(reward_rollout, torch.pow(torch.full((reward_rollout.shape[-1], ), discount_gamma, dtype=torch.float32, device=device), torch.arange(0, reward_rollout.shape[-1], device=device))).unsqueeze(-1)
            value_est = agent.value_network(start_obs)

            advantage = (action_value_est - value_est).squeeze(-1)

        p_dist = torch.distributions.Categorical(logits = agent.policy_network(start_obs))
        log_prob = p_dist.log_prob(action)

        # with torch.no_grad():
        #     old_p_dist = torch.distributions.Categorical(logits= agent.old_policy_network(start_obs))
        #     old_log_prob = old_p_dist.log_prob(action)
        #     ratio = torch.exp(log_prob - old_log_prob)

        #     clip_loss_first_term = ratio * advantage
        #     clip_loss_second_term = torch.clamp(ratio, 1. - clip_epsilon, 1. + clip_epsilon) * advantage
        
        # loss = -log_prob * torch.min(clip_loss_first_term, clip_loss_second_term) - entropy_coef * p_dist.entropy()


        with torch.no_grad():
            old_p_dist = torch.distributions.Categorical(logits= agent.old_policy_network(start_obs))
            old_log_prob = old_p_dist.log_prob(action)
        ratio = torch.exp(log_prob - old_log_prob)

        clip_loss_first_term = ratio * advantage
        clip_loss_second_term = torch.clamp(ratio, 1. - clip_epsilon, 1. + clip_epsilon) * advantage
    
        loss = -torch.min(clip_loss_first_term, clip_loss_second_term) - entropy_coef * p_dist.entropy()

        optimizer.zero_grad()
        loss.mean().backward()
        print(loss.mean().cpu().detach())
        optimizer.step()

def critic_trainer(agent: PPOAgent, expBuffer: ExperienceBuffer, optimizer: torch.optim.Optimizer, T_step: int, batch_size: int, num_runs: int = None, discount_gamma: float = 0.99):
    # when should a loop end? lets try using this simple method.
    if num_runs == None:
        num_runs = len(expBuffer) // batch_size

    critic_loss_fn = torch.nn.MSELoss()
    # loop start
    for i in range(num_runs):
        start_obs, end_obs, n_reward, action, terminated = map(torch.tensor, expBuffer.sample(batch_size, T_step))
        start_obs = start_obs.float().to(device)
        end_obs = end_obs.float().to(device)
        n_reward = n_reward.float().to(device)
        action = action.float().to(device)
        terminated = terminated.float().to(device)


        value_s = agent.value_network(start_obs)

        with torch.no_grad():
            reward_rollout = torch.concat([n_reward, agent.value_network(end_obs) * (1 - terminated)], dim=-1)

            action_value_est = torch.matmul(reward_rollout, torch.pow(torch.full((reward_rollout.shape[-1], ), discount_gamma, dtype=torch.float32, device=device), torch.arange(0, reward_rollout.shape[-1], device=device))).unsqueeze(-1)
            value_est = agent.value_network(start_obs)

            advantage = (action_value_est - value_est)

        loss = critic_loss_fn(value_s, advantage)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# env = gym.make("LunarLander-v2")
env = gym.make("LunarLander-v2", render_mode="human")
writer = SummaryWriter('LunarLandar_PPO')
model_save_path = "PPO.th"

n_step = 1000000

batch_size = 64
num_epochs = 3
T_step = 64
training_set_size = T_step * 8

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

print(state_shape)
print(action_shape)

observation, info = env.reset(seed=42)
env.render()

agent = PPOAgent_torch(state_shape, action_shape)
if os.path.exists(os.path.join(os.getcwd(), model_save_path)):
    agent.load(model_save_path)

buffer = ExperienceBuffer()
policy_optimizer = torch.optim.Adam(agent.policy_network.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(agent.value_network.parameters(), lr=3e-4)

episode = 0
total_reward = 0

for i in range(n_step):

    # generate state
    state = torch.tensor(observation).to(device)

    with torch.no_grad():
        action = agent.sample_action(state).cpu().detach().numpy()

    next_observation, reward, terminated, truncated, info = env.step(action)
    buffer.push(observation, action, reward, terminated)

    observation = next_observation
    total_reward += reward

    if len(buffer) >= training_set_size:
        # run train loop
        for _ in range(num_epochs):
            critic_trainer(agent, buffer, critic_optimizer, T_step, batch_size)
        for _ in range(num_epochs):
            PPO_trainer(agent, buffer, policy_optimizer, T_step, batch_size, clip_epsilon=0.1)
        
        # update old policy network after training is finished
        agent.update_old_policy_network()
        # clear buffer after training
        buffer.clear()

    if terminated or truncated:
        observation, info = env.reset()
        # buffer.clear()
        episode += 1

    if episode % 10 == 4:
        writer.add_scalar('average reward per episode', total_reward / 10, episode)
        total_reward = 0
    if episode % 100 == 99:
        agent.save(model_save_path)

env.close()

