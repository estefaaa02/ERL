# Import libaries for the speech to text and text preprocessing
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def normalize(str):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        (".",""),
        (",",""),
        (";",""),
        (":", "")
    )
    for a, b in replacements:
        str = str.replace(a, b).replace(a.upper(), b.upper())
    return str

#Method to identify the negations
def pottsBySentence(sentence):
  negations = ("no", "tampoco", "nadie", "jamás", "ni", "sin", "nada", "nunca", "ningún", "ninguno", "ninguna")
  words = sentence.split()
  for index, word in enumerate(words):
    if word in negations:
      for wordToChangeIndex in range(index+1, len(words)):
        if words[wordToChangeIndex] not in negations:
          if '_NOT' in words[wordToChangeIndex]:
            words[wordToChangeIndex]=words[wordToChangeIndex].replace('_NOT', '')
          else:
            words[wordToChangeIndex]+='_NOT'
  return ' '.join(map(str, words))

def pottsAlgorithm(textString):
  sentences = textString.split('.')
  for index, sentence in enumerate(sentences):
    sentences[index] = pottsBySentence(sentence)
  return sentences

def audio_to_text_dataset(corpus_file, dataset_dir):
  """
  This functions converts the audio to text using the dataset directory.
  
  It returns a dataframe with the text and its given emotion

  ARGUMENTS:
    -corpus_file: The path to the corpus.txt which contains the names of the audio files.
    -dataset_dir: The path to the directory that contains the audio files
  """
  # Define a variable and assign an instance of the recognizer class to it
  r = sr.Recognizer()
  # Opens the file that contains the names of the audio files
  f = open(corpus_file, "r")
  # Creates arrays to persist the file names, text of the audio and the emotion of said audio
  nombresArchivos = []
  contenidosArchivos = []
  emociones = []
  # Iterates through the lines of the corpus.txt file to get the names of the audio files
  for linea in f:
    # It only takes the files that are in spanish
    if "es" in linea:
      # If the line contains a line break, it is removed. Otherwise, the whole line is passed to the AudioFile class
      if(linea[-1]== "\n"):
        # The file is passed to the class AudioFile, which does a necessary preprocessing step to avoid errors related to the file's data type
        harvard = sr.AudioFile(dataset_dir + linea[:-1])
      else:
        # The file is passed to the class AudioFile, which does a necessary preprocessing step to avoid errors related to the file's data type
        harvard = sr.AudioFile(dataset_dir + linea)
      with harvard as source:
        # The record method is called to convert the audio file to an audio data type
        audio = r.record(source)
        try:
          #Uses Google's free web search API to do the speech to text
          texto = r.recognize_google(audio, language="es-CO")
        except:
          continue
      # Esta parte va dentro del loop que se encarga de leer los archivos
      # If the name of the audio file contains "gio" then the emotion of the audio is positive,
      # If it contains "dis" then it's neutral, otherwise the emotion is negative
      if "gio" in linea:
        emociones.append("pos")
      elif "dis" in linea:
        emociones.append("neu")
      else:
        emociones.append("neg")
        # The name of the file and the text that was obtained are saved in their respective arrays
      nombresArchivos.append(linea)
      contenidosArchivos.append(texto.lower())
  f.close()

  # A dataframe is created with the contents saved in the arrays
  data = {
          'archivo':nombresArchivos,
          'contenido': contenidosArchivos,
          "emociones": emociones
  }
  df = pd.DataFrame(data)
  return df

def preprocess_text_dataset(corpus_file, dataset_dir):
  """
  This function preprocess the text of the dataset.
  
  It returns the preprocessed text, its tags and the vocabulary size

  ARGUMENTS:
    -corpus_file: The path to the corpus.txt which contains the names of the audio files.
    -dataset_dir: The path to the directory that contains the audio files
  """
  # Define a variable and assign an instance of the recognizer class to it
  r = sr.Recognizer()
  # Transforms the audio to text and assigns it to a dataframe
  df = audio_to_text_dataset(corpus_file, dataset_dir)

  """
  This section does the preprocessing for the previously obtained text
  """
  stop_words = set(stopwords.words('spanish')) 
  stemmer = SnowballStemmer('spanish')
  lemmatizer = WordNetLemmatizer()
  preprocessed_text = []
  tags = []
  # Iterates through the dataframe
  for index, row in df.iterrows():
    # The sentence of the audio file is normalized using the method created
    normalized = normalize(row['contenido'])
    sentences = pottsAlgorithm(normalized)
    content = ''
    for sentence in sentences:
      content += sentence + ' '
    # The normalized sentence is tokenized and saved in an array
    token = nltk.word_tokenize(content)
    # Filters through the tokens and the stop words are taken out
    filtered_sentence = [w for w in token if not w in stop_words]
    lemmatized = []
    for word in filtered_sentence:
      # Each word is stemmed and lemmatized
      stemmed_word = stemmer.stem(word)
      lemmatized_word = lemmatizer.lemmatize(stemmed_word)
      lemmatized.append(lemmatized_word)
    # The lemmatized words are then saved in another array with its respective emotion on another array
    for t in lemmatized:
      preprocessed_text.append(t)
      tags.append(row['emociones'])

  # Calculates the vocabulary size
  vocab_size = len(set(preprocessed_text)) + 1
  # Does the label encoding for the preprocessed text and its tags
  label_encoder_text = LabelEncoder()
  label_encoder_tags = LabelEncoder()
  preprocessed_text = label_encoder_text.fit_transform(preprocessed_text)
  tags = label_encoder_tags.fit_transform(tags)
  # Saves the preprocessed text classes of the label encoding
  np.save('ERL/classes/text_classes.npy', label_encoder_text.classes_)
  # Transforms the tags to categorical based on the number of class names
  tags = to_categorical(tags, 3)

  return preprocessed_text, tags, vocab_size


def process_audio(filename):
  """
  This function preprocess a single audio.

  It returns the preprocessed text.

  ARGUMENTS:
    -filename: The path of the file to preprocess
  """
  # Define a variable and assign an instance of the recognizer class to it
  r = sr.Recognizer()
  harvard = sr.AudioFile(filename)
  with harvard as source:
    # The record method is called to convert the audio file to an audio data type
    audio = r.record(source)
    try:
      # Uses Google's free web search API to do the speech to text
      texto = r.recognize_google(audio, language="es-CO")
      """
      This section does the preprocessing for the previously obtained text
      """
      stop_words = set(stopwords.words('spanish')) 
      stemmer = SnowballStemmer('spanish')
      lemmatizer = WordNetLemmatizer()
      # The sentence of the audio file is normalized using the method created
      normalized = normalize(texto)
      sentences = pottsAlgorithm(normalized)
      content = ''
      for sentence in sentences:
        content += sentence + ' '
      # The normalized sentence is tokenized and saved in an array
      token = nltk.word_tokenize(content)
      # Filters through the tokens and the stop words are taken out
      filtered_sentence = [w for w in token if not w in stop_words]
      lemmatized = []
      for word in filtered_sentence:
        # Each word is stemmed and lemmatized
        stemmed_word = stemmer.stem(word)
        lemmatized_word = lemmatizer.lemmatize(stemmed_word)
        lemmatized.append(lemmatized_word)
      final_text = []
      #The lemmatized words are then saved in another array with its respective emotion on another array
      for t in lemmatized:
        #The preprocessed text is transformed to numerical values
        encoder = LabelEncoder()
        encoder.classes_ = np.load('ERL/classes/text_classes.npy')
        txt = encoder.transform([t])
        final_text.append(txt)
        
      return final_text[0]
    except Exception as e:
      print("error in audio", e.args)