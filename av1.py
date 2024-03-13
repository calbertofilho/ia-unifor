import numpy as np
import matplotlib.pyplot as plt

def perturb(x, e):
    return np.random.uniform(low=x-e, high=x+e)

def f(x1, x2):
    # Problema 1 (mínimo)
    return (x1 ** 2 + x2 ** 2)
    # Problema 2 (máximo)
    # return (np.exp(-(x1 ** 2 + x2 ** 2)) + 2 * np.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2)))
    # Problema 3 (mínimo)
    # return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
    # Problema 4 (mínimo)
    # return ((x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10))
    # Problema 5 (máximo)
    # return (((x1 * np.cos(x1)) / 20) + 2 * np.exp(-(x1 ** 2) - ((x2 - 1) ** 2)) + 0.01 * x1 * x2)
    # Problema 6 (máximo)
    # return ((x1 * np.sin(4 * np.pi * x1)) - (x2 * np.sin((4 * np.pi * x2) + np.pi)) + 1)
    # Problema 7 (mínimo)
    # return ((-np.sin(x1) * np.sin((x1 ** 2)/np.pi) ** (2 * 10)) - (np.sin(x2) * (np.sin((2 * x2 ** 2)/np.pi) ** (2 * 10))))
    # Problema 8 (mínimo)
    # return ((-(x2 + 47)) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47))))) - (x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47)))))

# Problema 1
x = np.linspace(-100, 100, 1000)
# Problema 2
# x = np.linspace(start=[-2, -2], stop=[4, 5], num=1000, axis=1)
# Problema 3
# x = np.linspace(-8, 8, 1000)
# Problema 4
# x = np.linspace(-5.12, 5.12, 1000)
# Problema 5
# x = np.linspace(-10, 10, 1000)
# Problema 6
# x = np.linspace(-1, 3, 1000)
# Problema 7
# x = np.linspace(0, np.pi, 1000)
# Problema 8
# x = np.linspace(-200, 20, 1000)

# Problemas 1, 3, 4, 5, 6, 7, 8
X1, X2 = np.meshgrid(x, x)
# Problema 2
# X1, X2 = np.meshgrid(x[0], x[1])
Y = f(X1, X2)

# Problema 1
x_otimo = np.random.uniform(-100, 100, size=(2, ))
# Problema 2
# x_otimo = np.random.uniform(low=[-2, -2], high=[4, 5], size=(2, ))
# Problema 3
# x_otimo = np.random.uniform(-8, 8, size=(2, ))
# Problema 4
# x_otimo = np.random.uniform(-5.12, -5.12, size=(2, ))
# Problema 5
# x_otimo = np.random.uniform(-10, 10, size=(2, ))
# Problema 6
# x_otimo = np.random.uniform(-1, 3, size=(2, ))
# Problema 7
# x_otimo = np.random.uniform(0, np.pi, size=(2, ))
# Problema 8
# x_otimo = np.random.uniform(-200, 20, size=(2, ))
f_otimo = f(x_otimo[0], x_otimo[1])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='x', s=90, linewidth=2, color='red')

ax.set_xlabel('valores x')
ax.set_ylabel('valores y')
ax.set_zlabel('valores z')
ax.set_title('f(x1, x2)')
# Problema 1
ax.view_init(elev=10., azim=-65., roll=0.)
# Problema 2
# ax.view_init(elev=24., azim=-66., roll=0.)
# Problema 3
# ax.view_init(elev=30., azim=-65., roll=0.)
# Problema 4
# ax.view_init(elev=25., azim=-61., roll=0.)
# Problema 5
# ax.view_init(elev=15., azim=-140., roll=0.)
# Problema 6
# ax.view_init(elev=30., azim=-60., roll=0.)
# Problema 7
# ax.view_init(elev=26., azim=-65., roll=0.)
# Problema 8
# ax.view_init(elev=30., azim=160., roll=0.)
plt.colorbar(surface)
plt.tight_layout()

e = .5
max_iter = 1000
max_vizinhos = 30
i = 0
melhoria = True

while i < max_iter and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        x_cand = perturb(x_otimo, e)
        f_cand = f(x_cand[0], x_cand[1])
        if(f_cand < f_otimo):
            melhoria = True
            x_otimo = x_cand
            f_otimo = f_cand
#            ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='x', s=20, linewidth=2, color='k')
            break
    i+=1

ax.text(x_otimo[0], x_otimo[1], f_otimo, "x = "+"{:.4f}".format(x_otimo[0])+"\ny = "+"{:.4f}".format(x_otimo[1])+"\nz = "+"{:.4f}".format(f_otimo), color='red')
ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='x', s=90, linewidth=2, color='green')

# Problema 1
# plt.savefig('av1-problema1-funcao1.png')
# Problema 2
# plt.savefig('av1-problema1-funcao2.png')
# Problema 3
# plt.savefig('av1-problema1-funcao3.png')
# Problema 4
# plt.savefig('av1-problema1-funcao4.png')
# Problema 5
# plt.savefig('av1-problema1-funcao5.png')
# Problema 6
# plt.savefig('av1-problema1-funcao6.png')
# Problema 7
# plt.savefig('av1-problema1-funcao7.png')
# Problema 8
# plt.savefig('av1-problema1-funcao8.png')

plt.show()