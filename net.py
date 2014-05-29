import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import numpy as np
import pylab as plt
from sklearn import svm, cluster
import random
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

def load_data(subject_range, start=0, end=375, test=False):

    # list of all subjects
    x = []
    y = []

    # load subject data
    for subject in subject_range:

        # open train/test file
        if test:
            filename = 'test_17_23/test_subject%02d.mat' % subject
        else:
            filename = 'train_01_16/train_subject%02d.mat' % subject

        # load
        data = loadmat(filename, squeeze_me=True)

        # test = id/train = labels
        if test:
            subject_y = data['Id']
        else:
            subject_y = data['y']

        # z score and resize
        subject_x = data['X']
        subject_x = zscore(subject_x)
        subject_x = subject_x[:,:,start:end].copy()

        # append to subject list
        x.append(subject_x)
        y.append(subject_y)

    # make numpy arrays
    x = np.vstack(x)
    y = np.concatenate(y)

    return x, y

class ts_cluster(object):
    def __init__(self,num_clust):
        '''
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        '''
        self.num_clust=num_clust
        self.assignments={}
        self.centroids=[]

    def k_means_clust(self,data,num_iter,w,progress=False):
        '''
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure.
        '''
        self.centroids=random.sample(data,self.num_clust)

        for n in range(num_iter):
            if progress:
                print 'iteration '+str(n+1)
            #assign data points to clusters
            self.assignments={}
            for ind,i in enumerate(data):
                min_dist=float('inf')
                closest_clust=None
                for c_ind,j in enumerate(self.centroids):
                    if self.LB_Keogh(i,j,5)<min_dist:
                        cur_dist=self.DTWDistance(i,j,w)
                        if cur_dist<min_dist:
                            min_dist=cur_dist
                            closest_clust=c_ind
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust]=[]

            #recalculate centroids of clusters
            for key in self.assignments:
                clust_sum=0
                for k in self.assignments[key]:
                    clust_sum=clust_sum+data[k]
                self.centroids[key]=[m/len(self.assignments[key]) for m in clust_sum]


    def get_centroids(self):
        return self.centroids

    def get_assignments(self):
        return self.assignments

    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()

    def DTWDistance(self,s1,s2,w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW={}

        if w:
            w = max(w, abs(len(s1)-len(s2)))

            for i in range(-1,len(s1)):
                for j in range(-1,len(s2)):
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            if w:
                for j in range(max(0, i-w), min(len(s2), i+w)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            else:
                for j in range(len(s2)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

        return np.sqrt(DTW[len(s1)-1, len(s2)-1])

    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind,i in enumerate(s1):

            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2

        return np.sqrt(LB_sum)


if __name__ == "__main__":

    # load training data
    X, y = load_data(range(1,2), 125, 225)

    cv = 2

    print "Computing cross-validated accuracy for each channel."
    clf = LogisticRegression(random_state=0)
    score_channel = np.zeros(X.shape[1])
    for channel in range(X.shape[1]):
        print "Channel #", channel, "score:",
        X_channel = X[:,channel,:].copy()
        scores = cross_val_score(clf, X_channel, y, cv=cv, scoring='accuracy')
        score_channel[channel] = scores.mean()
        print score_channel[channel]

    best_channels = np.argsort(score_channel)

    """
    print "Best channel #", best_channels[-1], "Accuracy:", score_channel[best_channels[-1]]
    X_best_face = X[:,best_channels[-1],:][y==1].mean(0)
    X_best_scramble = X[:,best_channels[-1],:][y==0].mean(0)

    plt.plot(X_best_face, 'r-')
    plt.plot(X_best_scramble, 'b-')
    plt.show()

    print "Worst channel #", best_channels[0], "Accuracy:", score_channel[best_channels[0]]
    X_worst_face = X[:,best_channels[0],:][y==1].mean(0)
    X_worst_scramble = X[:,best_channels[0],:][y==0].mean(0)

    plt.plot(X_worst_face, 'r-')
    plt.plot(X_worst_scramble, 'b-')
    plt.show()
    """


    kmeans = ts_cluster(2)

    kmeans.k_means_clust(X[:,best_channels[-1],:], 10, w=None, progress=True)

    assignments = kmeans.get_assignments()

    X_best_face = X[:,best_channels[-1],:][assignments[1]].mean(0)
    X_best_scramble = X[:,best_channels[-1],:][assignments[0]].mean(0)

    plt.plot(X_best_face, 'r-')
    plt.plot(X_best_scramble, 'b-')
    plt.show()