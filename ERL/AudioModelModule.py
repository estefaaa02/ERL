# Import dependencies for the model
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures as aF
from ERL import AudioProcessModule
import numpy as np 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Declare the directories for the negative, positive, and neutral audio corpus
dirs = ['ERL/data/negative', 'ERL/data/positive', 'ERL/data/neutral']

def audio_svm_model():
  """

  This function generates, trains and saves the svm model using the given dataset

  """

  # Preprocesses the audios of the directories using the preprocess method done in the audio process module
  y, f = AudioProcessModule.preprocess_dataset(dirs)

  # Splits the dataset in 80% training and 20% test
  X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2,random_state=1800000) 

  # train the svm classifier
  model = SVC(kernel = 'rbf', C = 400000)
  model1 = model.fit(X_train, y_train)
  # Persists the trained model using pickle
  filename = 'ERL/models/audio_svm_model.sav'
  pickle.dump(model1, open(filename, 'wb'))
  model1.score(X_train,y_train)
  #predicts using the test part of the dataset
  predicted=model1.predict(X_test)
  # Prints the metrics of the model
  # Model Accuracy: The set of labels predicted for a sample match exactly the corresponding set of labels of the dataset
  print("Accuracy:",metrics.accuracy_score(y_test, predicted))
  # Model Precision: The ability of the classifier not to label as positive a sample that is negative
  print("Precision:",metrics.precision_score(y_test, predicted, average="weighted"))
  # Model Recall: The ability of the classifier to find all the positive samples
  print("Recall:", metrics.recall_score(y_test, predicted, average="weighted"))
  # Model F-Score: Weighted average of the precision and recall
  print("F-Score:", metrics.f1_score(y_test, predicted, average="weighted"))
