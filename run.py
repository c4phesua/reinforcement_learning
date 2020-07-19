from rf_model.dqn_model import DQN
import gym
from tensorflow.keras.layers import Dense
from tensorflow.keras import losses
from tensorflow.keras import optimizers


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = DQN(0.9, 0.9, 32, 4096)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(8, activation='tanh', input_shape=(2,)))
    agent.training_network.add(Dense(8, activation='relu'))
    agent.training_network.add(Dense(3, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error)

    op2 = optimizers.RMSprop(learning_rate=0.00025)
    agent.target_network.add(Dense(8, activation='tanh', input_shape=(2,)))
    agent.target_network.add(Dense(8, activation='relu'))
    agent.target_network.add(Dense(3, activation='linear'))
    agent.target_network.compile(optimizer=op2, loss=losses.mean_squared_error)
    agent.update_target_network()
    count = 0
    for ep in range(200):
        state = env.reset()
        done = False
        print(ep, '------------------', 'current epsilon: ', agent.epsilon_greedy.epsilon)
        while not done:
            env.render()
            action = agent.observe_on_training(state)
            state, reward, done, _ = env.step(action)
            print(ep, '-----------------------------------', reward)
            agent.take_reward(reward, state, done)
            agent.train_network(32, 4, 4)
            count += 1
            if count % 16 == 0:
                agent.update_target_network()
        agent.epsilon_greedy.decay(0.95, 0.01)
