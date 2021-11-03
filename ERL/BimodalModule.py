import pathlib
from ERL import AudioProcessModule
from ERL import TextProcessingModule
import pickle
from tensorflow.keras.models import load_model
# Current directory
HERE = pathlib.Path(__file__).resolve().parent

def get_audio_svm_metrics(filepath):
  """
  This function extracts the calculated metrics for the svm model from a given file and returns them.
  :param filepath: Path to the file that contains the metrics
  :return: accuracy, precision, recall and f1-score of the svm model.
  """
  audio_svm_metrics = open(filepath, "r")
  audio_svm_accuracy = 0
  audio_svm_precision = 0
  audio_svm_recall = 0
  audio_svm_fscore = 0

  for line in audio_svm_metrics:
    splitted_line = line.split(":")
    
    if "Accuracy" in splitted_line[0]:
      audio_svm_accuracy = float(splitted_line[1][:-1])
    elif "Precision" in splitted_line[0]:
      audio_svm_precision = float(splitted_line[1][:-1])
    elif "Recall" in splitted_line[0]:
      audio_svm_recall = float(splitted_line[1][:-1])
    elif "F-Score" in splitted_line[0]:
      audio_svm_fscore = float(splitted_line[1])

  return audio_svm_accuracy, audio_svm_precision, audio_svm_recall, audio_svm_fscore

def get_text_cnn_metrics(filepath):
  """
    This function extracts the calculated metrics for the cnn model from a given file and returns them.
    :param filepath: Path to the file that contains the metrics
    :return: accuracy, precision, recall and f1-score of the cnn model.
    """
  text_cnn_metrics = open(filepath, "r")
  text_cnn_accuracy = 0
  text_cnn_precision = 0
  text_cnn_recall = 0
  text_cnn_fscore = 0

  for line in text_cnn_metrics:
    splitted_line = line.split(":")
    
    if "Accuracy" in splitted_line[0]:
      text_cnn_accuracy = float(splitted_line[1][:-1])
    elif "Precision" in splitted_line[0]:
      text_cnn_precision = float(splitted_line[1][:-1])
    elif "Recall" in splitted_line[0]:
      text_cnn_recall = float(splitted_line[1][:-1])
    elif "F-Score" in splitted_line[0]:
      text_cnn_fscore = float(splitted_line[1])

  return text_cnn_accuracy, text_cnn_precision, text_cnn_recall, text_cnn_fscore

def get_prediction_svm(input_file):
  """
    This function predicts the emotion out of a given audio file using the previously saved svm model.
    The result of this method will be primitive, because it just returns the results of the model.

    ARGUMENTS:
      -input_file: The path of the file to predict from.
    """
  # Extracts the features of the given audio
  features = AudioProcessModule.preprocess_single_audio(input_file)
  # Loads the previously generated svm model
  model = pickle.load(open((HERE / "models/audio_svm_model.sav"), 'rb'))
  # Predicts the emotion
  predicted = model.predict(features)

  return predicted


def get_prediction_cnn(input_file):
  """
    This function predicts the emotion out of a given audio file using the previously saved cnn model.
    The result of this method will be primitive, because it returns the result of the model.

    ARGUMENTS:
      -audio_file: The path of the file to predict from
    """
  encoded_text = TextProcessingModule.process_audio(input_file)
  model = load_model(HERE / "models/modelo_texto.h5")
  predicted = model.predict(encoded_text)
  return predicted


def bimodal(input_file):
  """
  This function calculates a prediction using the svm and the cnn models.

  :param input_file: The path of the file to predict from
  :return: The calculated prediction. 0 -> Negative, 1 -> Positive, 2 -> Neutral
  """
  # Predicts the emotion using the svm and cnn model
  prediction_svm = get_prediction_svm(input_file)
  prediction_cnn = get_prediction_cnn(input_file)

  # Gets an average of the prediction from the list given by the svm model
  sum_predictions_svm = 0
  for i in prediction_svm:
    sum_predictions_svm += i

  average_prediction_svm = sum_predictions_svm / len(prediction_svm)

  # Gets the prediction from each position on the list given by the cnn model
  # If the number on the given position can be rounded to 1 then that emotion is the one to be returned
  negative = prediction_cnn[0][0]
  neutral = prediction_cnn[0][1]
  positive = prediction_cnn[0][2]

  # Gets the metrics from the files
  audio_accuracy, audio_precision, audio_recall, audio_fscore = get_audio_svm_metrics(
    HERE / "metrics/audio_svm_metrics.txt")
  text_accuracy, text_precision, text_recall, text_fscore = get_text_cnn_metrics(HERE / "metrics/text_cnn_metrics.txt")

  # Calculates an average between the metrics
  average_metrics_svm = (audio_accuracy + audio_precision + audio_recall + audio_fscore) / 4
  average_metrics_cnn = (text_accuracy + text_precision + text_recall + text_fscore) / 4

  # Calculates the weighted prediction depending on which emotion returned a number that is higher
  # than the other two emotions.
  if negative > positive and negative > neutral:
    cnn_weighted_prediction = (1 - negative) * average_metrics_cnn
  elif neutral > positive and neutral > negative:
    cnn_weighted_prediction = (neutral * 2) * average_metrics_cnn
  elif positive > negative and positive > neutral:
    cnn_weighted_prediction = positive * average_metrics_cnn

  svm_weighted_prediction = average_prediction_svm * average_metrics_svm
  # Calculates the best case scenario for the positive and neutral emotions, and the worst case scenario for
  # the negative emotion
  best_case_positive = average_metrics_cnn + average_metrics_svm
  best_case_neutral = (2 * average_metrics_cnn) + (2 * average_metrics_svm)
  worst_case_negative = (0.5 * average_metrics_cnn) + (0.33 * average_metrics_svm)

  # Sums the final two weighted predictions
  final_result = svm_weighted_prediction + cnn_weighted_prediction

  # Uses the ranges previously calculated to determine which emotion to return
  if final_result <= worst_case_negative:
    return 0
  elif final_result > worst_case_negative and final_result <= best_case_positive:
    return 1
  elif final_result > best_case_positive and final_result <= best_case_neutral:
    return 2
