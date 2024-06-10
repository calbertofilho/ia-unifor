import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

class Adaline:
    def __init__(self, data: pd.DataFrame) -> None:
        self.setDados(data)