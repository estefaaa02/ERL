from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures as aF
import numpy as np

# Set non-overlapping windows
m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05 

def preprocess_dataset(dirs):
  """
  This function preprocesses the audios in the given directories.

  The result is the numpy array of the class names and the features that were extracted using pyAudioAnalysis.

  ARGUMENTS:
    - dirs: An array of the directories of the dataset. Each directory represents a class name.
  """

  # segment-level feature extraction:
  features = [] 
  for d in dirs: # get feature matrix for each directory (class) 
    f, files, fn = aF.directory_feature_extraction(d, m_win, m_step, 
                                                  s_win, s_step) 
    features.append(f)

  # select features and create feature matrices for the three classes:
  f1 = np.array([features[0][:, fn.index('mfcc_1_mean')],
                features[0][:, fn.index('mfcc_2_mean')]])
  f2 = np.array([features[1][:, fn.index('mfcc_1_mean')],
                features[1][:, fn.index('mfcc_2_mean')]])
  f3 = np.array([features[2][:, fn.index('mfcc_1_mean')],
                features[2][:, fn.index('mfcc_2_mean')]])
  # transform class names
  y = np.concatenate((np.zeros(f1.shape[1]), np.ones(f2.shape[1]), np.full(f3.shape[1], 2))) 
  # Transpose the feature matrices
  f = np.concatenate((f1.T, f2.T, f3.T), axis = 0)

  return y, f

def preprocess_single_audio(input_file):
  """
  This function preprocess the contents of a given audio.

  It returns the extracted features of the audio for later prediction.
  
  ARGUMENTS:
    -input_file: The path of the audio file.
  """

  # Read audio data from file
  # (returns sampling freq ad)
  sampling_rate, signal = audioBasicIO.read_audio_file(input_file)
  # Converts the input signal to mono
  signal = audioBasicIO.stereo_to_mono(signal)
  # Checks if the audio file is too small
  if signal.shape[0] < float(sampling_rate)/5:
    print("The audio file is too small")
  
  # Extracts the mid and short features of the audio
  mid_features, short_features, names = aF.mid_feature_extraction(signal, sampling_rate, round(m_win * sampling_rate), round(m_step * sampling_rate), round(sampling_rate * s_win), round(sampling_rate * s_step))
  # Transposes the mid features
  mid_features = np.transpose(mid_features)
  # Selects the mfcc_1_mean and mfcc_2_mean features of the vector
  features = np.array([mid_features[:, names.index('mfcc_1_mean')],
                      mid_features[:, names.index('mfcc_2_mean')]])
  # Returns the transposed selected features
  return features.T