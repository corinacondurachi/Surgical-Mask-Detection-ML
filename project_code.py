from scipy.stats import skew
import librosa
import numpy as np
from scipy.io import wavfile as wav
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# function for extracting the features from the audio files
def get_features(path, i):
    y, sample_rate = librosa.load(str(path) + str(i) + '.wav', sr = None)
    feature1 = librosa.feature.mfcc(y, sr = sample_rate)
    feature2 = librosa.feature.spectral_contrast(y)[0]
    feature1_preprocessed = np.hstack((np.mean(feature1, axis=1), np.std(feature1, axis=1), skew(feature1, axis = 1)))
    feature2_preprocessed = np.hstack((np.mean(feature2), np.std(feature2), skew(feature2)))
    return np.hstack((feature1_preprocessed, feature2_preprocessed))


# preparing the train set by extracting features from the audio files situated in train folder
path_train = '/Users/corinacondurachi/Downloads/ml-fmi-23-2020/train/train/'
train_data = []
# train set
for i in range (100001,108001):
    
    x = get_features(path_train, i)
    train_data.append(x)
    
train_data = np.asarray(train_data)


# extracting the labels for the coresponding audio files (train_data)
l = [None] * 8000
f = open("/Users/corinacondurachi/Downloads/ml-fmi-23-2020/train.txt", "r")
line = f.readline()
while line:
    l[int(line[0:6])-100001] = int(line[11])
    # read the next line
    line = f.readline()
f.close()
train_data_labels = np.array(l)


# building the model and applying fit
from sklearn.svm import SVC
clf = SVC(C = 100)
clf.fit(train_data, train_data_labels)


# preparing the validation set by extracting features from the audio files situated in validation folder
path_validation = '/Users/corinacondurachi/Downloads/ml-fmi-23-2020/validation/validation/'
valid_data = []
# validation set
for i in range (200001,201001):
    
    x = get_features(path_validation, i)
    valid_data.append(x)
    
validation_data = np.asarray(valid_data)  


# extracting the labels for the coresponding audio files (validation_data)
l = [None] * 1000
f = open("/Users/corinacondurachi/Downloads/ml-fmi-23-2020/validation.txt", "r")
line = f.readline()
while line:
    l[int(line[0:6])-200001] = int(line[11])
    # read the next line
    line = f.readline()
f.close()
validation_labels = np.array(l)


# applying score for validation data
clf.score(validation_data, validation_labels)


# making predictions on the validation data
validation_predictions = []
validation_predictions = clf.predict(valid_data)


# preparing the test set by extracting features from the audio files situated in test folder
path_test = '/Users/corinacondurachi/Downloads/ml-fmi-23-2020/test/test/'
test_data = []
# validation set
for i in range (300001,303001):
    
    x = get_features(path_test,i)
    test_data.append(x)
    
test_data = np.asarray(test_data)  


# making predictions on the test data
predictions = []
predictions = clf.predict(test_data)


# writing predictions for the test data in a txt file
f = open("/Users/corinacondurachi/Downloads/ml-fmi-23-2020/test15.txt", "w")
contor = 0
f.write('name,label\n')
for i in range (300001,303001):
    f.write(str(i) + '.wav,')
    f.write(str(predictions[contor]))
    f.write('\n')
    contor += 1
f.close()


# calculating the acores accuracy on validation data
scores = cross_val_score(clf, validation_data, validation_labels, cv = 5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# printing a classification report for the validation data
print(classification_report(validation_labels, validation_predictions))


# printing confusion matrix for the validation data
print(confusion_matrix(validation_labels, validation_predictions))
