import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import  plotting

alpha = 0.5
gamma = 1
epsilon = 0.05
num_episodes = 5000
max_steps = 1000
n_planning_steps = [1, 10, 100]
n_experiments = 5


# create the environment for the snakes and ladders
class SnakesAndLadders:
    def __init__(self):
        """Create the Snakes & Ladders Board"""
        self.board = np.zeros(101)
        self.ladders = {2: 32, 12: 37, 36: 58, 20: 62, 64: 95, 68: 97}
        self.snakes = {35: 2, 25: 8, 48: 9, 75: 54, 78: 52, 93: 64, 99: 70}
        self.curr = 1

    def reset(self):
        """Reset the board when done"""
        self.board = np.zeros(101)
        self.curr = 1

    def step(self, action):
        """Take a step within the environment"""
        reward = -1  # for all transitions except
        # squares with snakes and ladders
        terminal = False  # player has not won
        self.curr += action  # take the number of steps player chose

        # if current state more than 100, we have to
        # move backwards instead.
        if self.curr > 100:
            back = self.curr - 100
            self.curr -= 2 * back

        # check if we landed on a snake or ladder
        # if snake, descend. if ladder, ascend.
        if self.curr in self.ladders.keys():
            self.curr = self.ladders[self.curr]
        if self.curr in self.snakes.keys():
            self.curr = self.snakes[self.curr]

        # check if current is terminal
        # NOTE: only terminal when exactly at 100
        if self.curr == 100:
            terminal = True
            reward = 1

        self.board[self.curr] += 1

        return self.curr, reward, terminal

    def showBoard(self):
        """Shows board and how many times agent has landed in the state"""
        print("Snakes and Ladders game board:")

        print(np.round(self.board[:100], 1).reshape(10, 10))

    def getCurrent(self):
        """returns current state"""
        return self.curr

    def setCurrent(self, curr):
        self.curr = curr


class Agent:
    def __init__(self, alpha, gamma, epsilon):
        """Declare agent variables"""
        self.Q = np.zeros((101, 6))  # 100 states X 6 actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = []
        self.V = np.zeros((101, 6))

    def reset(self):
        self.actions = []

    def roll_dice(self):
        """Generate a roll of the dice"""
        # NOTE: the dice will be an unfair die
        diceRoll = np.random.choice(range(1, 7))
        return diceRoll

    def ep_greedy(self, state):
        """follow epsilon greedy policy to choose action"""
        diceRoll = self.roll_dice()  # we roll the dice
        if np.random.random() < self.epsilon:
            # we choose the number of steps limited from the diceRoll
            action = np.random.randint(1, diceRoll + 1)
        else:
            action = np.argmax(self.Q[state]) + 1  # we do this, otherwise
            # the action returned is
            # one less than what we roll.

        self.actions.append(action)
        return action

    def updateQ(self, s, a, s_prime, r):
        self.Q[s, a] += self.alpha * (r + self.gamma * max(self.Q[s_prime]) - self.Q[s, a])

    def updateSar(self, s, a, s_prime, r):
        self.Q[s, a] += self.alpha * (r + self.gamma * s_prime - self.Q[s, a])

    def updateTD(self, s, s2, r):
        self.V[s] += self.V[s] + self.alpha * (r + self.gamma * self.V[s2] - self.V[s])

    def getQValues(self):
        return self.Q

    def getActions(self):
        return self.actions

    def resetQ(self):
        self.Q = np.zeros((101, 6))


def QLearning(num_episodes, env, agent):
    actionList = []

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for ep in tqdm(range(num_episodes)):
        pass
        # get current state from environment
        state = env.getCurrent()
        t=1
        while True:
            # using epsilon-greedy policy, get actionfrom Agent object
            action = agent.ep_greedy(state)
           # print(action)
            # take a step in the environment to get
            # subsequent state and reward
            state_prime, reward, terminal = env.step(action)
           # print(state_prime)
            if terminal:
                break
            t=t+1
            # Update statistics
            stats.episode_rewards[ep] += reward
            stats.episode_lengths[ep] = t

            # using Agent object, update its Q values
            agent.updateQ(state, action - 1, state_prime, reward)
            state = state_prime
        actionList.append(agent.getActions())
        agent.reset()
        env.reset()
    return actionList,stats


def SLearning(num_episodes, env, agent):
    actionList = []

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for ep in tqdm(range(num_episodes)):
        pass
        # get current state from environment
        state = env.getCurrent()
        t = 1
        while True:
            # using epsilon-greedy policy, get actionfrom Agent object
            action = agent.ep_greedy(state)
            # print(action)
            # take a step in the environment to get
            # subsequent state and reward
            state_prime, reward, terminal = env.step(action)
            if terminal:
                break
            t = t + 1
            # Update statistics
            stats.episode_rewards[ep] += reward
            stats.episode_lengths[ep] = t
            # using Agent object, update its Q values
            agent.updateSar(state, action - 1, state_prime, reward)
            state = state_prime
        actionList.append(agent.getActions())
        agent.reset()
        env.reset()
    return actionList,stats

def TDLearning(num_episodes, env, agent):
    actionList = []

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for ep in tqdm(range(num_episodes)):
        pass
        # get current state from environment
        state = env.getCurrent()
        t = 1
        while True:
            # using epsilon-greedy policy, get actionfrom Agent object
            action = agent.ep_greedy(state)

            state_prime, reward, terminal = env.step(action)
            if terminal:
                break
            t = t + 1
            # Update statistics
            stats.episode_rewards[ep] += reward
            stats.episode_lengths[ep] = t
            # using Agent object, update its Q values
            state = state_prime
            agent.updateTD(state, state_prime, reward)
        actionList.append(agent.getActions())
        agent.reset()
        env.reset()
    return actionList,stats


def testSAndL():
    SAndL = SnakesAndLadders()
    while True:
        action = np.random.randint(1, 7)
        state_prime, reward, terminal = SAndL.step(action)
        SAndL.showBoard()
        if terminal:
            break
    print("finish")
    SAndL.showBoard()


def testStep(s):
    SAndL = SnakesAndLadders()
    print('{:^15}{:^15}{:^15}{:^15}'.format('Input State', 'Dice Roll', 'Next State', 'Reward'))
    for i in range(1, 7):
        SAndL.setCurrent(s)
        state_prime, reward, terminal = SAndL.step(i)
        print('{:^15}{:^15}{:^15}{:^15}'.format(s, i, state_prime, reward))


def testRollDice():
    test_agent = Agent(0.5, 0.8, 0.05)
    freq = dict()
    for i in range(1000):
        roll = test_agent.roll_dice()
        if roll not in freq.keys():
            freq[roll] = 1
        else:
            freq[roll] += 1
    print('{:^15}{:^15}'.format("Dice Roll", "Frequency"))
    for roll, f in sorted(freq.items()):
        print('{:^15}{:^15}'.format(roll, f / 1000))


def plot_actionNum_vs_episodes(num_episodes, actionList, algoTitle, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.set(title='actionNum vs Episodes for ' + algoTitle)
    ax.set_xlabel('Num of Episodes')
    ax.set_ylabel('Number of Actions taken')
    # xticks = np.linspace(0, 1000-1, num=100)
    # xticklabels = ["%d" % x for x in np.arange(1000)]
    # ax.set_xticks(xticks, xticklabels)

    actionLen = []
    for i in actionList:
        actionLen.append(len(i))

    ax.plot(np.arange(num_episodes), actionLen)



def plot_heatmap_max_val(env, value, algoTitle, ax=None):
    """Generate heatmap showing maximum value at each state"""
    if ax is None:
        fig, ax = plt.subplots()

    if value.ndim == 1:
        value_max = np.reshape(value[:100], (10, 10))
    else:
        value_max = np.reshape(value[:100].max(axis=1), (10, 10))
    value_max = value_max[::-1, :]

    im = ax.imshow(value_max, aspect='auto', interpolation='none', cmap='afmhot')
    ax.set(title='Maximum value per state for ' + algoTitle)
    ax.set_xticks(np.linspace(0, 10 - 1, num=10))
    ax.set_xticklabels(["%d" % x for x in np.arange(10)])
    ax.set_yticks(np.linspace(0, 10 - 1, num=10))

    ax.set_yticklabels(["%d" % y for y in np.arange(
        0, 100, 10)][::-1])

    return im


if __name__ == '__main__':
    QLearningAgent = Agent(alpha, gamma, epsilon)
    snakesAndLadders = SnakesAndLadders()
    actionList,statsQ = QLearning(num_episodes, snakesAndLadders, QLearningAgent)

  #  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
  #  plot_heatmap_max_val(snakesAndLadders, QLearningAgent.getQValues(), 'QLearning', ax[0])
  #  plot_actionNum_vs_episodes(num_episodes, actionList, "QLearning", ax[1])

    SLearningAgent = Agent(alpha, gamma, epsilon)
    snakesAndLadders = SnakesAndLadders()
    actionList,statsS = SLearning(num_episodes, snakesAndLadders, SLearningAgent)


  #  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
  # plot_heatmap_max_val(snakesAndLadders, SLearningAgent.getQValues(), 'SARSA', ax[0])
  # plot_actionNum_vs_episodes(num_episodes, actionList, "SARSA", ax[1])


    SLearningAgent = Agent(alpha, gamma, epsilon)
    snakesAndLadders = SnakesAndLadders()
    actionList,statsTD = TDLearning(num_episodes, snakesAndLadders, SLearningAgent)

    #plt.show()
    print("QLearning")
    plotting.plot_episode_stats(statsQ)
    print("SARSA")
    plotting.plot_episode_stats(statsS)
    print("TD")
    plotting.plot_episode_stats(statsTD)