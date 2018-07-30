import numpy as np
import matplotlib.pyplot as plt

#DIV_{kl}
#using without smoothing techniques
#todo: add smoothing
def kl(Q,L):
    assert Q.shape == L.shape
    return np.sum(Q*(np.log(Q)-np.log(L)))

def js(Q,L):
    pass

def plot_dist(Q,P):
    kl1 = kl(Q,P) 
    kl2 = kl(P,Q) 
    print(kl(Q,P))
    print(kl(P,Q))
    plt.plot(Q) 
    plt.plot(P)
    plt.text(0.4, .82, r'kl(Q,P)=' + str(kl1))
    plt.text(0.4, .78, r'kl(P,Q)=' + str(kl2))
    plt.xlabel('$x$')
    plt.ylabel('$P(x)$')
    plt.ylim([0,1])
    plt.show()

if __name__ == '__main__':
    print('starting up')
    Q = np.linspace(0.01, 1.01, 10)
    P = np.linspace(0.01, 1.01 ,10) + np.random.uniform(low=0, high=100, size=10)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    plot_dist(Q,P) 
