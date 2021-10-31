from ERL import AudioProcessModule
from ERL import TextProcessingModule
from ERL import BimodalModule
import pickle
from tensorflow.keras.models import load_model
import pathlib

# Current directory
HERE = pathlib.Path(__file__).resolve().parent

def predict_emotion_audio_svm(input_file):
  """
  This function predicts the emotion out of a given audio file using the previously saved svm model.

  ARGUMENTS:
    -input_file: The path of the file to predict from.
  """
  # Extracts the features of the given audio
  features = AudioProcessModule.preprocess_single_audio(input_file)
  # Loads the previously generated svm model
  model = load_model_audio(HERE / "models/audio_svm_model.sav")
  # Predicts the emotion
  predicted = model.predict(features)[0]
  # Depending on the prediction the function returns Negative, Positive or Neutral
  if predicted == 0:
    return 'Negative'
  elif predicted == 1:
    return 'Positive'
  elif predicted == 2:
    return 'Neutral'

def predict_emotion_text_cnn(audio_file):
  """
  This function predicts the emotion out of a given audio file using the previously saved cnn model.

  ARGUMENTS:
    -audio_file: The path of the file to predict from
  """
  encoded_text = TextProcessingModule.process_audio(audio_file)
  model = load_model_text(HERE / "models/modelo_texto.h5")
  predicted = model.predict(encoded_text)
  negative = round(predicted[0][0])
  neutral = round(predicted[0][1])
  positive = round(predicted[0][2])
  if negative == 1:
    return "Negative"
  elif neutral == 1:
    return "Neutral"
  elif positive == 1:
    return "Positive"
  else:
    return "Prediction not found"

def predict_emotion_bimodal(audio_file):
  """
  This function predicts the emotion out of a given audio file using a bimodal approach

  ARGUMENTS:
    -audio_file: The path of the file to predict from
  """
  # Calculate which model to use
  result = BimodalModule.bimodal()
  emotion = ""
  
  if result == 0:
    # If result is 0 then the model to use will be svm
    emotion = predict_emotion_audio_svm(audio_file)
  elif result == 1:
    # If result is 1 then the model to use will be cnn
    emotion = predict_emotion_text_cnn(audio_file)

  return emotion

def load_model_audio(file_path):
  """
  This function loads a model from a given path using pickle and returns said model.

  ARGUMENTS:
    -file_path: The path of the model to load.
  """
  # Loads the previously saved model to make the prediction
  load_model = pickle.load(open(file_path, 'rb'))
  
  return load_model

def load_model_text(file_path):
  """
  This function loads a model from a file path.

  It returns the loaded model.
  ARGUMENTS:
    -file_path: Path to the model.
  """
  loaded_model = load_model(file_path)

  return loaded_model