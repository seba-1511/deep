import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from tools import load_data
from pre_processing import z_score
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    # load, constrain to .5secs after stimulus, concat and z score
    train_x, train_y = load_data(range(1,8))
    train_x = train_x[:,:,125:250].copy()
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
    train_x = z_score(train_x)

    # load test data and transform
    test_x, id_y = load_data(range(1,8), test=True)
    test_x = test_x[:,:,125:250].copy()
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1]*test_x.shape[2])
    test_x = z_score(test_x)

    # train and classify
    clf = LogisticRegression(random_state=0)
    clf.fit(train_x, train_y)
    predicted_y = clf.predict(test_x)

    # create submission file
    f = open('submission.csv', 'w')
    print >> f, "Id,Prediction"
    for i in range(len(predicted_y)):
        print >> f, str(id_y[i]) + "," + str(predicted_y[i])

    f.close()