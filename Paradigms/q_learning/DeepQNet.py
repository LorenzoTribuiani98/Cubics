import torch
import numpy as np
from collections import deque
import itertools
import random
from tqdm import tqdm
from Paradigms.utils.utils import ACTION_MAP, REVERSE_BLOCK_MAP, REVERSE_BLOCK_MAP_, create_new_state, compute_reward, check_lines


GAMMA = 0.9                   
BATCH_SIZE = 216  
BUFFER_SIZE = 50000           #size of the replay buffer
MIN_REPLAY_SIZE = 5000        #minumum number of element to start training
EPSILON_START = 1.0           #initial epsilon
EPSILON_END = 0.02            #final epsilon
EPSILON_DECAY = 30000         #number of step to decrease epsilon
TARGET_UPDATE_FREQ = 10000     #number of step to udpdate the target net
PRINT_INTERVAL = 1000         #number of step for printing
OPTIMAL_LEN = 5000            #size of optimal buffer

class Agent(torch.nn.Module):

    def __init__(self, hidden_layers=2, initial_neurons=512, *args, **kwargs) -> None:
        """
        crate a new model

        Parameters
        ----------

        - hidden_layer: number of hidden fully connected layers
        - initial_neurons: number of neurons of the first fully conected leayers
        """
        super().__init__(*args, **kwargs)

        self.conv1 = torch.nn.Conv2d(1,4, (2,1), 1)
        self.conv2 = torch.nn.Conv2d(4,8, (2,1), 1)
        self.conv3 = torch.nn.Conv2d(8,16,(2,1), 1)
        self.batchnorm1 = torch.nn.BatchNorm2d(4)
        self.batchnorm2 = torch.nn.BatchNorm2d(8)
        self.batchnorm3 = torch.nn.BatchNorm2d(16)

        self.max_pool = torch.nn.MaxPool2d((2,2))

        self.conv_block = torch.nn.Conv2d(1,2, (3,3), 1)
        self.batchnorm_b = torch.nn.BatchNorm2d(2)

        self.linear_in = torch.nn.Linear(18, initial_neurons)
        self.hiddens = [torch.nn.Linear(initial_neurons//int((2**(i+1))/2), initial_neurons//int((2**(i+2))/2)) for i in range(hidden_layers)]
        self.out = torch.nn.Linear(initial_neurons//int((2**(hidden_layers+1))/2), 20)

        self.activation = torch.nn.Tanh()

    def forward(self, field, block):
        field = field.reshape((-1,1,20,10))
        block = block.reshape((-1, 1, 3, 3))

        x_f = self.conv1(field)
        x_f = self.max_pool(x_f)
        x_f = self.activation(x_f)

        x_f = self.conv2(x_f)
        x_f = self.max_pool(x_f)
        x_f = self.activation(x_f)

        x_f = self.conv3(x_f)
        x_f = self.max_pool(x_f)
        x_f = self.activation(x_f)

        x_b = self.conv_block(block)
        x_b = self.activation(x_b)

        x_f = torch.flatten(x_f, start_dim=1)
        x_b = torch.flatten(x_b, start_dim=1)

        x = torch.cat((x_f, x_b), dim=1)

        x = self.linear_in(x)
        x = self.activation(x)
        for layer in self.hiddens:
            x = layer(x)
            x = self.activation(x)

        x = self.out(x)
        return x
    
    def act(self, field, blocks):
        field = torch.as_tensor(field).unsqueeze(0).float()
        blocks = torch.as_tensor(blocks).unsqueeze(0).float()
        q_values = self(field, blocks)
        action = torch.argmax(q_values, dim=1)[0].detach().item()

        return action

class DeepQNet():

    def __init__(self,
                 model_online,
                 model_target,
                 loss,
                 buffer_size = BUFFER_SIZE,
                 print_interval = PRINT_INTERVAL,
                 optimals_len = OPTIMAL_LEN,
                 minimum_replay_size = MIN_REPLAY_SIZE,
                 epsilon_decay = EPSILON_DECAY,
                 epsilon_start = EPSILON_START,
                 epsilon_end = EPSILON_END,
                 target_update_frequency = TARGET_UPDATE_FREQ) -> None:
        self.online_net = model_online
        self.target_net = model_target
        self.target_net.load_state_dict(self.target_net.state_dict())

        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=print_interval)
        self.loss_buffer = deque(maxlen=print_interval)
        self.optimals_buffer = deque(maxlen=optimals_len) 
        self.epsilon_decay = epsilon_decay  
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end 
        self.target_update_frequency = target_update_frequency   
        self.print_interval = print_interval 

        self.optimizer = torch.optim.RMSprop(self.online_net.parameters(), 5e-6, 0.9, momentum=0.9)
        self.loss = loss

        self.minimum_replay_size = minimum_replay_size

    def learn(self, positive_buffer = False, initial_blocks = 100, min_score=5, block_increment = 2, step_increment=7000, max_increment=1000):
        """
        Train the network for the deep q learning algorithm

        Parameters
        ----------

        - positive buffer: use the positive buffer
        - initial_blocks: number of blocks in a game
        - min_score: minimum reward to be inserted in the positive buffer
        - block_increment: number of blocks added to the initial blocks
        - step_increment: number of steps between teo blocks increment
        - max_increment: maximum number of blocks
        """
        episode_reward = 0
        blocks_elapsed = 0
        progress_bar = tqdm(range(self.minimum_replay_size))
        best_rew = float("-inf")

        field = np.zeros([20,10])
        blocks = np.random.randint(9, size=(2))
        counter = 0

        while counter <= MIN_REPLAY_SIZE:
            blocks_elapsed += 1
            action = np.random.randint(20)
            
            bad_placing = False
            if ACTION_MAP[action][0] + REVERSE_BLOCK_MAP_[blocks[0]][0] >= 10:
                bad_placing = True

            reward = compute_reward(field, blocks[0], ACTION_MAP[action], bad_placing, row_n=20)
            episode_reward += reward
            new_field, _ = create_new_state(field, blocks[0], ACTION_MAP[action])

            if new_field is None:
                new_blocks = np.random.randint(9, size=(2))
                new_features = (np.zeros([20,10]), REVERSE_BLOCK_MAP[new_blocks[0]])
                if not positive_buffer:
                    self.replay_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 1))
                    counter += 1
                    progress_bar.update()
                field = np.zeros([20,10])
                blocks = new_blocks
                self.reward_buffer.append(episode_reward)
                episode_reward = 0.0

            else:
                new_blocks = np.array([blocks[1], np.random.randint(9)])
                new_features = (new_field, REVERSE_BLOCK_MAP[new_blocks[0]])

                
                if positive_buffer and reward >=min_score:
                    self.optimals_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 0))
                    self.replay_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 0))
                    counter += 1
                    progress_bar.update()

                elif not positive_buffer:
                    self.replay_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 0))
                    counter += 1
                    progress_bar.update()

                if blocks_elapsed == initial_blocks:
                    field = np.zeros([20,10])
                    blocks = np.random.randint(9, size=2)
                    blocks_elapsed = 0
                    self.reward_buffer.append(episode_reward)
                    episode_reward = 0.0
                else:
                    field, _ = check_lines(new_field)
                    blocks = new_blocks
                             


        field = np.zeros([20,10])
        blocks = np.random.randint(9, size=(2))
        blocks_elapsed = 0
        for step in itertools.count():
            blocks_elapsed += 1
            epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

            if np.random.random() < epsilon:
                action = np.random.randint(20)
            else:
                action = self.online_net.act(field, REVERSE_BLOCK_MAP[blocks[0]])

            bad_placing = False
            if ACTION_MAP[action][0] + REVERSE_BLOCK_MAP_[blocks[0]][0] >= 10:
                bad_placing = True

            reward = compute_reward(field, blocks[0], ACTION_MAP[action], bad_placing, row_n = 20)
            episode_reward += reward
            new_field, _ = create_new_state(field, blocks[0], ACTION_MAP[action])

            if new_field is None:
                new_blocks = np.random.randint(9, size=(2))
                new_features = (np.zeros([20, 10]), REVERSE_BLOCK_MAP[new_blocks[0]])
                #self.replay_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 1))
                #if positive_buffer and len(self.optimals_buffer) > 0:
                #    self.replay_buffer.append(self.optimals_buffer[np.random.randint(len(self.optimals_buffer))]) # keep one positive element in list
                blocks = new_blocks
                self.reward_buffer.append(episode_reward)
                episode_reward = 0.0

            else:
                new_blocks = np.array([blocks[1], np.random.randint(9)])
                new_features = (new_field, REVERSE_BLOCK_MAP[new_blocks[0]])
                self.replay_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 0))
                if positive_buffer and reward >= min_score:
                    self.optimals_buffer.append(((field, REVERSE_BLOCK_MAP[blocks[0]]), action, reward, new_features, 0))
                if positive_buffer and reward < min_score:
                    self.replay_buffer.append(self.optimals_buffer[np.random.randint(len(self.optimals_buffer))]) # keep one positive element in list                        
                if blocks_elapsed == initial_blocks:
                    field = np.zeros([20,10])
                    blocks = np.random.randint(9, size=(2))
                    self.reward_buffer.append(episode_reward)
                    episode_reward = 0
                    blocks_elapsed = 0
                else:
                    field, _ = check_lines(new_field)
                    blocks = new_blocks
                

            timesteps = random.sample(self.replay_buffer, BATCH_SIZE)
            in_features = (torch.as_tensor(np.asarray([timestep[0][0] for timestep in timesteps]), dtype=torch.float32),
                        torch.as_tensor(np.asarray([timestep[0][1] for timestep in timesteps]), dtype=torch.float32))
            actions_batch = torch.as_tensor(np.asarray([timestep[1] for timestep in timesteps]), dtype=torch.int64).unsqueeze(-1)
            rewards_batch = torch.as_tensor(np.asarray([timestep[2] for timestep in timesteps]), dtype=torch.float32).unsqueeze(-1)
            new_features = (torch.as_tensor(np.asarray([timestep[3][0] for timestep in timesteps]), dtype=torch.float32),
                            torch.as_tensor(np.asarray([timestep[3][1] for timestep in timesteps]), dtype=torch.float32))
            dones_batch = torch.as_tensor(np.asarray([timestep[4] for timestep in timesteps]), dtype=torch.float32).unsqueeze(-1)

            target_q_values = self.target_net(new_features[0], new_features[1])
            max_target_qs = target_q_values.max(dim=1, keepdim=True)[0]

            targets = rewards_batch + GAMMA * (1 - dones_batch) * max_target_qs


            q_values = self.online_net(in_features[0], in_features[1])
            action_q_values = torch.gather(q_values, 1, actions_batch)

            loss = self.loss(action_q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_buffer.append(loss.item())

            
            if step % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            
            if step % step_increment == 0 and step != 0:
                if initial_blocks + block_increment <= max_increment:
                    initial_blocks += block_increment
                    print()
                    print(f"increasing blocks from {initial_blocks - block_increment} to {initial_blocks}")                
                

            if step % self.print_interval == 0:
                avg_rew = np.mean(self.reward_buffer)
                print()
                print("step          ", step)
                print("avg_rew:      ", avg_rew)
                print("loss:         ", np.mean(self.loss_buffer))
                print("mean_batch_r: ", np.mean(rewards_batch.cpu().numpy()))
                print("epsilon:      ", epsilon)

                if  avg_rew > best_rew and step >= 1000:
                    best_rew = avg_rew
                    torch.save(self.online_net.state_dict(), "online.pt")
                    torch.save(self.target_net.state_dict(), "target.pt")


