import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS= 10000
EPS=0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p = the win rate

        self.p = p
        self.p_estimate = 5.
        self.N = 1. # num of samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N-1)*self.p_estimate + x) / self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):

        # use optimistic initial values to select the next bandit
        j = np.argmax([b.p_estimate for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution of the bandit whose arm we just pulled
        bandits[j].update(x)


    for b in bandits:
        print('mean estimate:', b.p_estimate)

    # print total reward:
    print(f'Total reward earned:{rewards.sum()}')
    print(f'Overall win rate :{rewards.sum() / NUM_TRIALS}')

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) +1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == '__main__':
    experiment()
