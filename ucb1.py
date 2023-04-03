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
        self.N = 0. # num of samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N-1)*self.p_estimate + x) / self.N


def ucb(mean, n, nj):
    return mean + np.sqrt(2*np.log(n) / nj)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.empty(NUM_TRIALS)
    total_plays=0

    #init - play each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_TRIALS):

        # use optimistic initial values to select the next bandit
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution of the bandit whose arm we just pulled
        bandits[j].update(x)

        total_plays += 1

    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS)+1)

    for b in bandits:
        print('mean estimate:', b.p_estimate)

    # print total reward:
    print(f'Total reward earned:{rewards.sum()}')
    print(f'Overall win rate :{rewards.sum() / NUM_TRIALS}')

    # plot the results
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == '__main__':
    experiment()
