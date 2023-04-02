import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS= 10000
EPS=0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p = the win rate

        self.p = p
        self.p_estimate = 0.
        self.N = 0. # num of samples colelcted so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N-1)*self.p_estimate + x) / self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print('optimal j:', optimal_j)
    eps = EPS
    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit

        if np.random.random() < eps:
            num_times_explored +=1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited +=1
            j = np.argmax([b.p_estimate for b in bandits])

        if j==optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        rewards[i] = x

        bandits[j].update(x)



    for b in bandits:
        print('mean estimate:', b.p_estimate)

    # print total reward:
    print(f'Total reward earned:{rewards.sum()}')
    print(f'Overall win rate :{rewards.sum() / NUM_TRIALS}')
    print(f'Number of times explored:{num_times_explored}')
    print(f'Number of times exploited:{num_times_exploited}')
    print(f'Number of times selected optimal bandit:{num_optimal}')

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) +1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == '__main__':
    experiment()
