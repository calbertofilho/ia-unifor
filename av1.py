import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x1, x2):
    # return (x1 ** 2 + x2 ** 2)
    # return (np.exp(-(x1 ** 2 + x2 ** 2)) + 2 * np.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2)))
    # return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
    # return ((x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10))
    # return (((x1 * np.cos(x1)) / 20) + 2 * np.exp(-(x1 ** 2) - ((x2 - 1) ** 2)) + 0.01 * x1 * x2)
    # return ((x1 * np.sin(4 * np.pi * x1)) - (x2 * np.sin((4 * np.pi * x2) + np.pi)) + 1)
    # return ((-np.sin(x1) * np.sin((x1 ** 2)/np.pi) ** (2 * 10)) - (np.sin(x2) * np.sin((x2 ** 2)/np.pi) ** (2 * 10)))
    return ((-(x2 + 47)) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47))))) - (x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47)))))

# x1 = np.linspace(-100, 100, 1000)
# x1 = np.linspace(-2, 4, 1000)
# x1 = np.linspace(-8, 8, 1000)
# x1 = np.linspace(-5.12, 5.12, 1000)
# x1 = np.linspace(-10, 10, 1000)
# x1 = np.linspace(-1, 3, 1000)
# x1 = np.linspace(0, np.pi, 1000)
x1 = np.linspace(-200, 20, 1000)
X1,X2 = np.meshgrid(x1, x1)
Y = f(X1, X2)

# x1_cand, x2_cand = 50, 50
# x1_cand, x2_cand = 0, -1
# x1_cand, x2_cand = 0, -2
# x1_cand, x2_cand = 1, -5
# x1_cand, x2_cand = -2, 0
# x1_cand, x2_cand = -1, 0
# x1_cand, x2_cand = 3, 2
x1_cand, x2_cand = 3, 2
f_cand = f(x1_cand, x2_cand)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
ax.scatter(x1_cand, x2_cand, f_cand, marker='x', s=90, linewidth=3, color='red')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x1, x2)')
plt.tight_layout()

# plt.savefig('av1-problem1.png')
# plt.savefig('av1-problem2.png')
# plt.savefig('av1-problem3.png')
# plt.savefig('av1-problem4.png')
# plt.savefig('av1-problem5.png')
# plt.savefig('av1-problem6.png')
# plt.savefig('av1-problem7.png')
plt.savefig('av1-problem8.png')

#plt.show()