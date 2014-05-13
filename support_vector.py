import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import numpy as np
from sklearn import svm
from tools import load_data, constrain_interval, concatenate_sensors, z_score

# load and process training data
train_x, train_y = load_data(range(1,17))
train_x = constrain_interval(train_x, 0, 0.5)
train_x = concatenate_sensors(train_x)
train_x = z_score(train_x)

#valid_x, valid_y = load_data(range(9,17))
#valid_x = constrain_interval(valid_x, 0, 0.5)
#valid_x = concatenate_sensors(valid_x)
#valid_x = z_score(valid_x)

test_x, test_y = load_data(range(1,8), test=True)
test_x = constrain_interval(test_x, 0, 0.5)
test_x = concatenate_sensors(test_x)
test_x = z_score(test_x)

print "Training set:", train_x.shape

clf = svm.LinearSVC(verbose=True)
clf.fit(train_x, train_y)

print "Test set:", test_x.shape

y_pred = clf.predict(test_x)

filename_submission = "svm_submission.csv"
print "Creating submission file", filename_submission
f = open(filename_submission, "w")
print >> f, "Id,Prediction"
for i in range(len(y_pred)):
    print >> f, str(test_y[i]) + "," + str(y_pred[i])

f.close()
