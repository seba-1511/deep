import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pybrain
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
from sklearn import svm

def load_data(subject_range, start=0, end=375):

    train_x = []
    train_y = []

    for subject in subject_range:

        filename = 'train_01_16/train_subject%02d.mat' % subject
        data = loadmat(filename, squeeze_me=True)
        x = data['X']
        y = data['y']
        x = zscore(x)
        x = x[:,:,start:end].copy()

        train_x.append(x)
        train_y.append(y)

    train_x = np.vstack(train_x)
    train_y = np.concatenate(train_y)

    return train_x, train_y

if __name__ == "__main__":

    sensor_ranking = [266,286,284,262,218,277,238,283,194,215,275,217,290,226,280,221,302,294,249,146,268,269,14,64,190,246,127,153,270,156,229,11,253,166,301,304,170,267,293,299,151,175,297,143,195,188,300,70,169,173,79,147,159,185,256,39,97,57,4,28,88,254,120,214,289,240,15,2,8,67,243,6,130,178,191,244,128,138,181,235,101,161,279,85,95,123,187,202,38,23,164,150,303,210,44,197,305,223,263,168,17,18,133,9,184,114,136,232,98,186,7,131,34,193,274,0,3,41,140,265,291,233,248,292,261,139,250,241,142,108,132,199,213,100,111,116,192,42,58,252,259,75,83,134,227,158,25,154,251,285,296,109,287,203,46,162,271,272,5,91,115,207,148,77,80,93,94,122,167,171,247,45,10,31,117,245,295,62,107,201,163,22,105,196,258,29,55,61,78,37,60,180,224,19,32,145,209,65,189,103,137,74,155,236,49,183,219,119,27,212,182,30,110,257,68,71,124,152,288,149,172,281,125,35,92,104,273,24,234,81,86,89,96,76,112,222,231,264,59,82,205,72,121,16,179,36,47,204,211,43,87,176,113,102,21,200,66,160,84,282,255,56,99,144,230,90,276,12,63,298,40,260,13,157,33,118,53,69,48,174,206,220,52,242,50,126,135,141,225,106,198,177,73,129,20,54,278,237,51,1,26,165,239,216,208,228]

    for sensors in range(5, 306, 5):

        train_x, train_y = load_data(range(1,8),125,225)
        valid_x, valid_y = load_data(range(9,17),125,225)

        train_x = train_x[:,sensor_ranking[0:sensors],:].copy()
        train_x = train_x.reshape(train_x.shape[0],train_x.shape[1]*train_x.shape[2])

        valid_x = valid_x[:,sensor_ranking[0:sensors],:].copy()
        valid_x = valid_x.reshape(valid_x.shape[0],valid_x.shape[1]*valid_x.shape[2])

        linear_svm = svm.LinearSVC()
        linear_svm.fit(train_x, train_y)

        print "Sensors:", sensors, "LinearSVC:", linear_svm.score(valid_x, valid_y)