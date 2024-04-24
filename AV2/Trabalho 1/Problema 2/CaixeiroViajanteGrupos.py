import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("CaixeroGrupos.csv", names=["X", "Y", "Z", "G"])

def createNavigationMap(data, dominio):
    x = np.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
    X1, X2 = np.meshgrid(x[0], x[1])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(data.tail(1).index.item() +1):
        if data.at[i, 'G'] == 0:
            ax.scatter(data.at[i, 'X'], data.at[i, 'Y'], data.at[i, 'Z'], marker='*', s=50, linewidth=1, color='green')
        elif data.at[i, 'G'] == 1:
            color = 'blue'
        elif data.at[i,'G'] == 2:
            color = "gray"
        elif data.at[i,'G'] == 3:
            color = "purple"
        elif data.at[i,'G'] == 4:
            color = "orange"
        if data.at[i, 'G'] != 0:
            ax.scatter(data.at[i, 'X'], data.at[i, 'Y'], data.at[i, 'Z'], marker=('*' if i == data.tail(1).index.item() else 'o'), s=(50 if i == data.tail(1).index.item() else 10), linewidth=1, color=('red' if i == data.tail(1).index.item() else color))
    ax.set_title('Caixeiro Viajante Tridimensional Simples')
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    plt.tight_layout()
    ax.view_init(elev=30, azim=-65, roll=0.)
    plt.show()

def run() -> None:
    data = get_data()
    limites = [(data.iloc[:, 0].min(), data.iloc[:, 0].max()), (data.iloc[:, 1].min(), data.iloc[:, 1].max())]
    createNavigationMap(data, limites)

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        run()
finally:
    close()
