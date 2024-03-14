import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

def perturb(x, e, dominio):
    res = np.random.uniform(low=x-e, high=x+e, size=(2, ))
    if res[0] < dominio[0][0]:
        res[0] = dominio[0][0]
    elif res[0] > dominio[0][1]:
        res[0] = dominio[0][1]
    if res[1] < dominio[1][0]:
        res[1] = dominio[1][0]
    elif res[1] > dominio[1][1]:
        res[1] = dominio[1][1]
    return res

def f(x1, x2):
    # Problema 1 (mínimo)
    # return (x1 ** 2 + x2 ** 2)
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
    return ((-(x2 + 47)) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47))))) - (x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47)))))

# Domínio da função
# Problema 1
# dom = [(-100, 100), (-100, 100)]
# Problema 2
# dom = [(-2, 4), (-2, 5)]
# Problema 3
# dom = [(-8, 8), (-8, 8)]
# Problema 4
# dom = [(-5.12, 5.12), (-5.12, 5.12)]
# Problema 5
# dom = [(-10, 10), (-10, 10)]
# Problema 6
# dom = [(-1, 3), (-1, 3)]
# Problema 7
# dom = [(0, np.pi), (0, np.pi)]
# Problema 8
dom = [(-200, 20), (-200, 20)]

# Geração do grid da função
x = np.linspace(start=[dom[0][0], dom[1][0]], stop=[dom[0][1], dom[1][1]], num=1000, axis=1)
X1, X2 = np.meshgrid(x[0], x[1])
Y = f(X1, X2)

# Geração do ponto inicial
x_otimo = np.random.uniform(low=[dom[0][0], dom[1][0]], high=[dom[0][1], dom[1][1]], size=(2, ))
f_otimo = f(x_otimo[0], x_otimo[1])

# Desenho do gráfico e do ponto inicial
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='x', s=90, linewidth=2, color='red')

# Etiquetas dos eixos
ax.set_xlabel('valores x')
ax.set_ylabel('valores y')
ax.set_zlabel('valores z')
ax.set_title('f(x1, x2)')

# Problema 1
# ax.view_init(elev=10., azim=-65., roll=0.)
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
ax.view_init(elev=30., azim=160., roll=0.)
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
        x_cand = perturb(x_otimo, e, dom)
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