import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from NeuralNetwork import NeuralNetwork
from NeuralNetwork import NeuralNetworkTarget

class Agent(object):
    def __init__(self, alpha, beta,  tau, env, input_dims=2, gamma=0.99,
                 n_actions=2, max_size=1000000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size
        self.network = NeuralNetwork(alpha)
        self.target = NeuralNetworkTarget(beta)

        # self.scale = 1.0
        # self.noise = np.random.normal(scale=self.scale, size=(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):       # q-value??
        # self.network.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.network.device)
        mu = self.network.forward(observation).to(self.network.device)
        # mu_prime = mu + T.tensor(self.noise(),
        #                          dtype=T.float).to(self.actor.device)
        # self.network.train()
        return mu.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):   # store data in the buffer
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.target.device)
        done = T.tensor(done).to(self.target.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.target.device)
        action = T.tensor(action, dtype=T.float).to(self.target.device)
        state = T.tensor(state, dtype=T.float).to(self.target.device)

        self.network.optimizer.zero_grad()
        mu = self.network.forward(state, action)

        # self.target_actor.eval()
        # self.target_critic.eval()
        self.target.eval()
        target_value_ = self.target.forward(new_state,action)
        # critic_value_ = self.target_critic.forward(new_state, target_actions)
        # critic_value = self.critic.forward(state, action)

        target = []

        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * target_value_[j] * done[j])
        target = T.tensor(target).to(self.target.device)
        target = target.view(self.batch_size, 1)

        self.network.train()
        self.network.optimizer.zero_grad()
        network = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() +
            (1 - tau) * target_critic_dict[name].clone()


        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() +
            (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)