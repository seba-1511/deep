from tools import load_data
import numpy as np

def average(x):
    averaged_x = [(trial.mean(axis=0)) for trial in x]
    averaged_x = np.array(averaged_x)
    return averaged_x

if __name__ == "__main__":
    x, y = load_data(range(1,3))

    averaged_x = average(x)
    print averaged_x.shape