import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable
from agent import Agent
from memory.memory import Memory
from utils import soft_update_target_network, hard_update_target_network
from utils.noise import OrnsteinUhlenbeckActionNoise
from torchsummary import summary

class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,),
                 output_layer_init_std=None, activation="selu", **kwargs):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state.to(torch.float16), action_parameters.to(torch.float16)), dim=1)#沿着dim=1维拼接
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "selu":
                x = F.selu(self.layers[i](x.to(torch.float32)))
            elif self.activation == "mish":
                x = F.mish(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        Q = self.layers[-1](x).to(torch.float16)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="selu", init_std=None):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state.to(torch.float32)
        # print("state ",state)
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "selu":
                x = F.selu(self.layers[i](x))
            elif self.activation == "mish":
                x = F.mish(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        #for i in range(0, num_hidden_layers):
            #if self.activation == "relu":
                #x = F.relu(self.layers[i](x))
            #elif self.activation == "leaky_relu":
                #x = F.leaky_relu(self.layers[i](x), negative_slope)
            #else:
                #raise ValueError("Unknown activation function "+str(self.activation))
        # print("x ", x)
        action_params = self.action_parameters_output_layer(x)
        # print("state in ", self.action_parameters_passthrough_layer(state.to(torch.float32)))
        action_params += self.action_parameters_passthrough_layer(state.to(torch.float32))
        action_params = action_params.to(torch.float16)
        # print("action param ", action_params)

        if self.squashing_function:
            assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params * self.action_param_lim
        # action_params = action_params / torch.norm(action_params) ## REMOVE --- normalisation layer?? for pointmass
        return action_params


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 mode="train",
                 loss_func=F.mse_loss, # F.mse_loss
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1,self.num_actions+1)])
        self.action_parameter_size = np.int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        # print([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)])
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = 0.01
        self.epsilon_steps = 1000
        self.indexed = False
        self.weighted = False
        self.average = False
        self.random_weighted = False

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = 16
        self.gamma = 0.95
        self.replay_memory_size = 100000
        self.initial_memory_threshold = 500
        self.learning_rate_actor = 0.001
        self.learning_rate_actor_param = 0.0001
        self.inverting_gradients = True
        self.tau_actor = 0.1
        self.tau_actor_param = 0.001
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = 10.
        self.zero_index_gradients = False

        self.np_random = None
        self.seed = seed
        self._seed(seed)
        self.mode = mode

        self.use_ornstein_noise = True
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001) #, theta=0.01, sigma=0.01)

        # print(self.num_actions+self.action_parameter_size)
        self.replay_memory = Memory(self.replay_memory_size, observation_space.shape, (1+self.action_parameter_size,), next_actions=False)
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        if torch.cuda.device_count() > 10:
            print("使用{}个GPU训练".format(torch.cuda.device_count()))
            self.actor = nn.DataParallel(self.actor, device_ids=[0, 1])
            self.actor_target = nn.DataParallel(self.actor_target, device_ids=[0, 1])
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        if torch.cuda.device_count() > 10:
            print("使用{}个GPU训练".format(torch.cuda.device_count()))
            self.actor_param = nn.DataParallel(self.actor_param, device_ids=[0, 1])
            self.actor_param_target = nn.DataParallel(self.actor_param_target, device_ids=[0, 1])
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        if torch.cuda.device_count() > 10:
            passthrough_layer = self.actor_param.module.action_parameters_passthrough_layer
        else:
            passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        # print(initial_weights.shape)
        # print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            # print(initial_bias.shape)
            # print(passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def act(self, state, remain_order, remain_order_today, np_avaiable_action_zero_today):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            if torch.cuda.device_count() > 10:
                all_action_parameters = self.actor_param.module.forward(state)
            else:
                all_action_parameters = self.actor_param.forward(state)
            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            if self.mode == "train" and rnd < self.epsilon:
                action = self.np_random.choice(remain_order_today)#从[0, self.num_actions)中随机输出一个随机数
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                              self.action_parameter_max_numpy))
                print("rnd action ", action)
            else:
                # select maximum action
                if torch.cuda.device_count() > 10:
                    Q_a = self.actor.module.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))#state.unsqueeze(0),表示在第0维增加1维度，state的shape由[9]变为[1,9]
                else:
                    Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))#state.unsqueeze(0),表示在第0维增加1维度，state的shape由[9]变为[1,9]
                Q_a = Q_a.detach().cpu().data.numpy()  #Q_a shape is [1,action_size]
                Q_a[0][np_avaiable_action_zero_today == -1] = -float("inf")
                action_temp = np.argmax(Q_a[0])
                action = np_avaiable_action_zero_today[action_temp]
                print("qmax act ", action)
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            #print(" all_action_parameters",  all_action_parameters)
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=np.int).sum()
            if self.mode == "train" and self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            action_parameters = all_action_parameters[offset:offset+self.action_parameter_sizes[action]]
            # print(action_parameters)
            # print(self.actor_param[offset:offset+self.action_parameter_sizes[action]])
            
            # print("actionparam ", action_parameters)

        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            # print("index ", index.float(), " max_p ", max_p, " min_p ", min_p, " rnge ", rnge)
            # print("~index ", (~index).float())
            # print("vals ", vals)
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]
            # print("grad index ", grad[index])
            # print("grad index ", grad[~index])
            # print("grad ", grad)

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal, remain_order, np_avaiable_action_zero_today, done_action_update):
        act, all_action_parameters = action
        self._step += 1

        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        self._add_sample(state, np.concatenate(([act],all_action_parameters)).ravel(), reward, next_state, np.concatenate(([next_action[0]],next_action[1])).ravel(), \
                         remain_order, done_action_update, terminal=terminal)
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss(np_avaiable_action_zero_today)
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, remain_order, done_action_update, terminal):
        assert len(action) == 1 + self.action_parameter_size
        # if done_action_update[0] == 1.:
        #     self.replay_memory.clear_done_action(remain_order)
        #     done_action_update = [0.]
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self, np_avaiable_action_zero_today):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            pred_Q_a[0][np_avaiable_action_zero_today == -1] = torch.tensor(-65000).to(torch.float16).to(self.device)
            Qprime = torch.max(pred_Q_a.to(torch.float32), 1, keepdim=True)[0].squeeze()#取最大Q值，squeeze()：移除数组中维度为1的维度
            # print("Qprime ", Qprime, " terminals ", terminals)
            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime
        # Compute current Q-values using policy network
        # print("state ", states.cpu().numpy())
        if (np.isnan(states.cpu().numpy())).all():
            print("8.71states ", states.cpu().numpy())
            #print("statestop stop stop stop")
            exit()
        # print("action_parameters ", action_parameters.cpu().numpy())
        if (np.isnan(action_parameters.cpu().numpy())).all():
            print("8.72action_parameters ", action_parameters.cpu().numpy())
            #print("action_parametersstop stop stop stop")
            exit()
        q_values = self.actor(states, action_parameters)
        if (np.isnan(q_values.cpu().detach().numpy())).all():
            print("state ", states.cpu().numpy())
            print("action_parameters ", action_parameters.cpu().numpy())
            print("8.8q_values ", q_values.cpu().detach().numpy())
            #print("stop stop stop stop")
            exit()
        # print("action shape ", actions.view(-1, 1), " q value ", q_values)
        #print("11112222stop stop stop stop")
        y_predicted = q_values.to(torch.float32).gather(1, actions.view(-1, 1)).squeeze()#y_predicted为action对应的Q value
        # print("y_predicted ", y_predicted)
        y_expected = target.to(torch.float32)
        loss_Q = self.loss_func(y_predicted, y_expected)
        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        assert (self.weighted ^ self.average ^ self.random_weighted) or \
               not (self.weighted or self.average or self.random_weighted)
        Q = self.actor(states, action_params)
        Q_val = Q
        # print("before q value ", Q_val)
        if self.weighted:#false
            # approximate categorical probability density (i.e. counting)
            counts = Counter(actions.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:#false
            Q_val = Q / self.num_actions
        elif self.random_weighted:#false
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
        if self.indexed:
            Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))#整个batch的Q_val相加后，求均值
            # print("mean q value ", Q_loss)
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        # print("action_params ", action_params)
        delta_a = deepcopy(action_params.grad.data)
        # print("data a", delta_a, " shape ", delta_a.shape)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)
        # print("data a", delta_a, " shape ", delta_a.shape)
        # print("action_params ", action_params)
        out = -torch.mul(delta_a, action_params)#对两个张量进行逐元素乘法
        # print("out ", out, " shape ", out.shape)
        # print("actor param ", list(self.actor_param.parameters()))
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + 'mpdqn_qnet.pt')
        torch.save(self.actor_param.state_dict(), prefix + 'mpdqn_actor_param_net.pt')
        # print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + 'mpdqn_qnet.pt', map_location=self.device))
        self.actor_param.load_state_dict(torch.load(prefix + 'mpdqn_actor_param_net.pt', map_location=self.device))
        print("load dir ", prefix + 'mpdqn_actor_param_net.pt')
        # print('Models loaded successfully')
    def save_target_models(self, prefix):
        torch.save(self.actor_target.state_dict(), prefix + 'mpdqn_target_qnet.pt')
        torch.save(self.actor_param_target.state_dict(), prefix + 'mpdqn_target_actor_param_net.pt')
        
    def load_target_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor_target.load_state_dict(torch.load(prefix + 'mpdqn_target_qnet.pt', map_location=self.device))
        self.actor_param_target.load_state_dict(torch.load(prefix + 'mpdqn_target_actor_param_net.pt', map_location=self.device))
        print("load dir ", prefix + 'mpdqn_target_actor_param_net.pt')
         