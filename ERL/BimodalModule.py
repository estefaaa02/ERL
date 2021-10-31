import pathlib

# Current directory
HERE = pathlib.Path(__file__).resolve().parent

AUDIO_ACCURACY_NORMALIZATION = 0.15
AUDIO_PRECISION_NORMALIZATION = 0.12
AUDIO_RECALL_NORMALIZATION = 0.12
AUDIO_FSCORE_NORMALIZATION = 0.15

TEXT_ACCURACY_NORMALIZATION = 0.1
TEXT_PRECISION_NORMALIZATION = 0.08
TEXT_RECALL_NORMALIZATION = 0.1
TEXT_FSCORE_NORMALIZATION = 0.1
  

def get_audio_svm_metrics(filepath):
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

def bimodal():
  audio_accuracy, audio_precision, audio_recall, audio_fscore = get_audio_svm_metrics(HERE / "metrics/audio_svm_metrics.txt")
  text_accuracy, text_precision, text_recall, text_fscore = get_text_cnn_metrics(HERE / "metrics/text_cnn_metrics.txt")

  audio_accuracy = audio_accuracy * AUDIO_ACCURACY_NORMALIZATION
  audio_precision = audio_precision * AUDIO_PRECISION_NORMALIZATION
  audio_recall = audio_recall * AUDIO_RECALL_NORMALIZATION
  audio_fscore = audio_fscore * AUDIO_FSCORE_NORMALIZATION

  text_accuracy = text_accuracy * TEXT_ACCURACY_NORMALIZATION
  text_precision = text_precision * TEXT_PRECISION_NORMALIZATION
  text_recall = text_recall * TEXT_RECALL_NORMALIZATION
  text_fscore = text_fscore * TEXT_FSCORE_NORMALIZATION

  result = 0
  if audio_accuracy > text_accuracy and audio_precision > text_precision and audio_recall > text_recall and audio_fscore > text_fscore:
    result = 0
  elif audio_accuracy > text_accuracy and audio_precision > text_precision and audio_recall > text_recall and audio_fscore > text_fscore:
    result = 1
  elif audio_accuracy > text_accuracy and audio_precision > text_precision and audio_recall > text_recall:
    result = 0
  elif audio_accuracy < text_accuracy and audio_precision < text_precision and audio_recall < text_recall:
    result = 1
  elif audio_precision > text_precision and audio_recall > text_recall:
    result = 0
  elif audio_precision < text_precision and audio_recall < text_recall:
    result = 1
  elif audio_accuracy > text_accuracy:
    result = 0
  elif audio_accuracy < text_accuracy:
    result = 1

  return result