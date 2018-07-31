import numpy as np
import matplotlib.pyplot as plt

#DIV_{kl}
def kl(Q,L):
    assert Q.shape == L.shape
    return np.sum(Q*(np.log(Q)-np.log(L)))
#smoothed version of KL
#M = 1/2(Q+L)
# js = 0.5 * kl(Q|M) + 0.5*kl(L|M)
def js(Q,L):
    M = 0.5 * (Q+L)
    return 0.5 * (kl(Q,M) + kl(L,M))

def cross_entropy(Q,L):
    return -1 * np.sum(np.dot(Q,np.log(L)))


def plot_dist(Q,P):
    kl1 = kl(Q,P) 
    kl2 = kl(P,Q) 
    cr_ent = cross_entropy(Q,P)
    js1 = js(Q,P) 
    js2 = js(P,Q) 
    print(kl(Q,P))
    print(kl(P,Q))
    plt.plot(Q) 
    plt.plot(P)
    plt.text(0.4, .82, r'kl(Q,P)=' + str(kl1))
    plt.text(0.4, .78, r'kl(P,Q)=' + str(kl2))
    plt.text(0.4, .74, r'js(Q,P)=' + str(js1))
    plt.text(0.4, .70, r'js(P,Q)=' + str(js2))
    plt.text(0.4, .66, r'cross_entropy(P,Q)=' + str(cr_ent))
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
