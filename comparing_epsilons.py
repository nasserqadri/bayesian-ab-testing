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
        return np.random.randn() +  self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N-1)*self.p_estimate + x) / self.N


def experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    means = np.array([m1, m2, m3])
    true_best = np.argmax(means)
    count_suboptimal = 0

    data = np.empty(N)


    for i in range(N):

        # use epsilon-greedy to select the next bandit

        if np.random.random() < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.p_estimate for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        bandits[j].update(x)

        if j != true_best:
            count_suboptimal += 1

        data[i] =x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)


    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print('mean estimate:', b.p_estimate)

    print('Percent suboptimal for epsilon: %s:' % eps, float(count_suboptimal)/N)

    return cumulative_average



if __name__ == '__main__':
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_1 = experiment(m1, m2, m3, 0.1, 100000)
    c_05 = experiment(m1, m2, m3, 0.05, 100000)
    c_01 = experiment(m1, m2, m3, 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label='eps=0.1')
    plt.plot(c_05, label='eps=0.05')
    plt.plot(c_01, label='eps=0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps=0.1')
    plt.plot(c_05, label='eps=0.05')
    plt.plot(c_01, label='eps=0.01')
    plt.legend()
    plt.show()