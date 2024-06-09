import os
import sys
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
from pathlib import Path
from algoritmos.mlp import MLP
from algoritmos.adaline import Adaline
from algoritmos.perceptron import Perceptron

def load_full_data(file_name: str, columns: list, separator: str, ignore_header: bool) -> pd.DataFrame:
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dados"))
    if ignore_header:
        return pd.read_csv(file_name, names=columns, sep=separator, skiprows=[0])
    return pd.read_csv(file_name, names=columns, sep=separator)

def run():
    espiral = load_full_data(file_name="spiral.csv", columns=["x1", "x2", "y",], separator=",", ignore_header=False)
    aerogerador = load_full_data("aerogerador.dat", ["Vel", "Pot"], "\t", False)
    red_wine = load_full_data("winequality-red.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
    white_wine = load_full_data("winequality-white.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
    data = aerogerador
    perc = Perceptron(data)
    print(perc.getDados())
    print(perc.getShape())

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        run()
finally:
    close()