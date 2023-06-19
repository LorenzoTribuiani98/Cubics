from Q_table import train
from DeepQNet import Agent, DeepQNet
import torch

if __name__ == '__main__':

    # Q-table
    #----------------------------------
    #train(num_workers=8)

    # DeepQNet
    #----------------------------------
    online = Agent()
    target = Agent()
    loss = torch.nn.MSELoss()

    agent = DeepQNet(online, target, loss)
    agent.learn(positive_buffer=True, initial_blocks=5, min_score=5, max_increment=30, step_increment=10000, block_increment=1)