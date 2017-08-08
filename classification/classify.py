# Using k_nearest_neighbors algorithm to classify data in seeds dataset

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    file = open(filename, 'r')
    data = file.readlines()
    dataset = []
    for line in data:
      line = line.split('\n')[0]
      line = line.split('\t')
      line = [float(e) for e in line if len(e) > 0]
      dataset.append(line)
    return dataset

def knn_train(features, labels, _n_neighbors, _n_folds, normalize=False):
    classifier = KNeighborsClassifier(n_neighbors=_n_neighbors)
    if normalize:
        classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
    kf = KFold(len(features), n_folds=_n_folds, shuffle=True)
    acc_arr = []
    for training, testing in kf:
        training_features = [features[i] for i in training]
        training_labels = [labels[i] for i in training]
        classifier.fit(training_features, training_labels)
        testing_features = [features[i] for i in testing]
        testing_labels = [labels[i] for i in testing]
        predictions = classifier.predict(testing_features)
        cur_acc = np.mean(predictions == testing_labels)
        acc_arr.append(cur_acc)
    mean_acc = np.mean(acc_arr)
    return classifier, acc_arr, mean_acc

dataset = load_data('seeds_dataset.txt')

features = [row[:len(row) - 1] for row in dataset]

labels = [row[-1] for row in dataset]

# Classify without normalizing features
classifier, acc_arr, mean_acc = knn_train(features, labels, 5, 5, normalize=False)
acc_arr = ["{:.2%}".format(e) for e in acc_arr]
print ("Classify without normalizing features: ")
print ("\t K-fold accuracy: ", acc_arr)
print ("\t Mean accuracy: {:.2%}".format(mean_acc))

# Classify using normalizing features
classifier, acc_arr, mean_acc = knn_train(features, labels, 5, 5, normalize=True)
acc_arr = ["{:.2%}".format(e) for e in acc_arr]
print ("Classify using normalizing features: ")
print ("\t K-fold accuracy: ", acc_arr)
print ("\t Mean accuracy: {:.2%}".format(mean_acc))
