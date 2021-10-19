# Import libaries for the speech to text, text preprocessing and the classification
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
import tensorflow.keras.backend as K
from ERL import TextProcessingModule

def cnn_model(xtrain, ytrain, vocab_size):
  """
  This method defines, trains, and saves the cnn model.

  It returns the trained model.

  ARGUMENTS:
    -xtrain: The text to train the model with
    -ytrain: The label of the text to traintte mooel with
    -vocab_size: Size of the vocabulary
  """

  # Define model
  model = Sequential()
  model.add(Embedding(vocab_size, 100))
  model.add(Conv1D(128, 1, activation='relu'))
  model.add(GlobalMaxPooling1D())
  model.add(Flatten())
  model.add(Dense(50, activation='relu'))
  model.add(Dense(40, activation='relu'))
  model.add(Dense(20))
  model.add(Dense(15))
  model.add(Dense(12, activation='relu'))
  model.add(Dense(12, activation='relu'))
  model.add(Dense(3, activation='softmax'))
  print(model.summary())
  # Compile network
  model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['accuracy', Precision(), Recall()])
  # Fit network
  model.fit(xtrain, ytrain, epochs=500, batch_size=512)
  # Saves the model
  filename = 'ERL/models/modelo_texto.h5'
  model.save(filename)

  return model

def get_f1(y_true, y_pred):
  """
  This function calculates the f1-score metric.

  It returns the f1 value.
  ARGUMENTS:
    -y_true: The correct label of the prediction.
    -y_pred: The predicted labels.
  """
  y_true = tensorflow.cast(y_true, tensorflow.float32)
  y_pred = tensorflow.cast(y_pred, tensorflow.float32)
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val

def train_cnn_model():
  """
  This function trains the cnn model using the dataset.
  """
  # Preprocesses the text of the dataset
  preprocessed_text, tags, vocab_size = TextProcessingModule.preprocess_text_dataset("ERL/data/es/corpus.txt", "ERL/data/es/")

  # The previously processed text is separated and assigned to 
  # variables for the training and testing of the model
  x_train, x_test, y_train, y_test = train_test_split(preprocessed_text, tags, test_size=0.25, random_state=1) 
  # Trains the cnn model
  model = cnn_model(preprocessed_text, tags, vocab_size)

  # Evaluates the model using Accuracy, Precision and Recall
  loss, acc, precision, recall = model.evaluate(x_test, y_test, verbose=1, )
  print('Test Accuracy: %f' % (acc*100))
  print('Test Precision: %f' % (precision*100))
  print('Test Recall: %f' % (recall*100))
  # Evaluates the model using F1-Score
  predicted = model.predict(x_test)
  fscore = get_f1(y_test, predicted)
  print("Test F1-Score: %f" % (get_f1(y_test, predicted)*100))

  f = open("ERL/metrics/text_cnn_metrics.txt", "w")
  f.write("Accuracy:" + str(acc))
  f.write("\nPrecision:" + str(precision))
  f.write("\nRecall:" + str(recall))
  f.write("\nF-Score:" +str(fscore.numpy()))
  f.close()

