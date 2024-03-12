import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(10, 2)

plt.scatter(x[:, 0], x[:, 1], color='blue', edgecolors='k')
plt.show()