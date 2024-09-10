from scipy.optimize import differential_evolution
import numpy as np


def run_fn(values):
    print(values.shape)

bound = [[0, 1.0], [-99, 99]]
# bounds = [bound for _ in range(5)]
differential_evolution(run_fn, bound)
