from tools import load_data
from tools import butterfly_plot
import numpy as np

def z_score(x):
    x -= x.mean(0)
    x = np.nan_to_num(x / x.std(0))
    return x

def average(x):
    averaged_x = [(trial.mean(axis=0)) for trial in x]
    averaged_x = np.array(averaged_x)
    return averaged_x

def euclidean_distance(x):

    print "hi"

def dynamic_time_warping_distance(x):

    print "bye"

if __name__ == "__main__":
    x, y = load_data(range(1,2))

    x = z_score(x)

    averaged_x = average(x)
    print averaged_x.shape

    butterfly_plot(averaged_x[:10], y[:10])