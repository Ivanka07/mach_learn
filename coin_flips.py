import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
print 'Starting up'
#what is that
dist = stats.beta
#n_trials = np.arange(0,500, 50)
#n_trials = np.concatenate(([2, 3, 5, 7], n_trials))
n_trials = [0,1,2,3,4,5,8,15,50,500]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0,1,100)

for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials)/2, 2, k+1)
    plt.xlabel("$p$, probability of heads")
    #what is it
    plt.setp(sx.get_yticklabels(), visible=False)
    print(data[:N])
    heads = data[:N].sum()
    print 'Heads', heads
    y = dist.pdf(x, 1 + heads, 1 + N - heads)
    print 'y', y
    plt.plot(x, y, label = 'observe %d tosses, %d heads' %(N, heads))
    plt.fill_between(x,0, y, color='#348ABD', alpha=0.4) 
    #plt.vlines(0.5, 0.4, color='k', linestyles='--', lw=1)

    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)

plt.suptitle('Bayesian updating of posterios props', y=1.02, fontsize=14)
plt.show()
