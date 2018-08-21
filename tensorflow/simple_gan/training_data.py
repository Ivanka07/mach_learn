import numpy as np
import math
import matplotlib.pyplot as plt

def generate_training_data(size, plot_data = False):
    result = []
    for theta in range(0, size):
        theta = float(theta/100.0)
        x = math.cos(theta)
        y = math.sin(theta)
        xend = x * -1.0
        yend = y * -1.0  
        sample = np.array([np.linspace(x,xend,10), np.linspace(y,yend,10)], dtype=np.float32).reshape(20,1)
        result.append(sample)
    print('Generatd %s lines' % len(result)) 
    if plot_data:
        for i in range(0, len(result)):
            l = result[i]
            plt.plot(l[0],l[1], '*')
        plt.show()
    return result

def get_batches(training_data, batch_size):
    training_np = np.asarray(training_data)
    num_batches = len(training_data)/batch_size
    batches = [] 
    for batch in range(0, int(num_batches)):
        batches.append(training_np[batch:batch+batch_size, :,:])
    return np.asarray(batches)

data = generate_training_data(618, plot_data = False)
batches = get_batches(data, 30)

