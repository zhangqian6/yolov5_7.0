import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SiLU(x):
  ''' It returns x times sigmoid(x). '''
  return x * sigmoid(x)
x = np.linspace(-10, 10)
plt.plot(x, SiLU(x))
plt.axis('tight')
plt.title('Activation Function : SiLU')
plt.show()
