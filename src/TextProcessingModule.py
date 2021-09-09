#Import libaries for the speech to text, text preprocessing and the classification
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

"""
This section takes a zip file which contains the corpus and extracts all the audio files from it
"""
from zipfile import ZipFile
import os
  
# specifying the zip file name
archivo_dir = os.path.dirname(__file__)
file_name = archivo_dir + "\\corpus.zip"
  
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall(archivo_dir)
    print('Done!')

#Method to normalize the words. Any special characters are replaced
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

"""
This sections creates the methods for identifying the negated words
"""
def pottersBySentence(sentence):
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

def pottersAlgotithm(textString):
  sentences = textString.split('.')
  for index, sentence in enumerate(sentences):
    sentences[index] = pottersBySentence(sentence)
  print(sentences)
#Define a variable and assign an instance of the recognizer class to it
r = sr.Recognizer()
#Opens the file that contains the names of the audio files
f = open(archivo_dir + "\\es\\corpus.txt", "r")
#Creates arrays to persist the file names, text of the audio and the emotion of said audio
nombresArchivos = []
contenidosArchivos = []
emociones = []
#Iterates through the lines of the corpus.txt file to get the names of the audio files
for linea in f:
  #It only takes the files that are in spanish
  if "es" in linea:
    #If the line contains a line break, it is removed. Otherwise, the whole line is passed to the AudioFile class
    if(linea[-1]== "\n"):
      #The file is passed to the class AudioFile, which does a necessary preprocessing step to avoid errors related to the file's data type
      harvard = sr.AudioFile(archivo_dir + "\\es\\" + linea[:-1])
    else:
      #The file is passed to the class AudioFile, which does a necessary preprocessing step to avoid errors related to the file's data type
      harvard = sr.AudioFile(archivo_dir + "\\es\\" + linea)
    with harvard as source:
      #The record method is called to convert the audio file to an audio data type
      audio = r.record(source)
      try:
        #Uses Google's free web search API to do the speech to text
        texto = r.recognize_google(audio, language="es-CO")
      except:
        continue
    #Esta parte va dentro del loop que se encarga de leer los archivos
    #If the name of the audio file contains "gio" then the emotion of the audio is positive, otherwise the emotion is negative
    if "gio" in linea:
      emociones.append("pos")
    else:
      emociones.append("neg")
      #The name of the file and the text that was obtained are saved in their respective arrays
    nombresArchivos.append(linea)
    contenidosArchivos.append(texto.lower())
f.close()


#A dataframe is created with the contents saved in the arrays
data = {
        'archivo':nombresArchivos,
        'contenido': contenidosArchivos,
        "emociones": emociones
}
df = pd.DataFrame(data)

"""
This section does the preprocessing for the previously obtained text
"""
stop_words = set(stopwords.words('spanish')) 
stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()
preprocessed_text = []
tags = []
#Iterates through the dataframe
for index, row in df.iterrows():
  #The sentence of the audio file is normalized using the method created
  normalized = normalize(row['contenido'])
  #The normalized sentence is tokenized and saved in an array
  token = nltk.word_tokenize(normalized)
  #Filters through the tokens and the stop words are taken out
  filtered_sentence = [w for w in token if not w in stop_words]
  lemmatized = []
  for word in filtered_sentence:
    #Each word is stemmed and lemmatized
    stemmed_word = stemmer.stem(word)
    lemmatized_word = lemmatizer.lemmatize(stemmed_word)
    lemmatized.append(lemmatized_word)
  #The stemmed and lemmatized words are then tagged
  #tagged = nltk.pos_tag(lemmatized)
  #The tagged words are then saved in another array with its respective emotion on another array
  for t in lemmatized:
    #The preprocessed text is transformed to numerical values
    print(t,label_encoder.fit_transform([t]))
    txt = label_encoder.fit_transform([t])
    preprocessed_text.append(t)
    emotion = [row['emociones'], row['emociones']]
    print(emotion)
    tags.append(row['emociones'])

vocab_size = len(set(preprocessed_text))
preprocessed_text = label_encoder.fit_transform(preprocessed_text)
tags = label_encoder.fit_transform(tags)

def cnn_model(xtrain, ytrain, xtest, ytest, vocab_size):

  # define model
  model = Sequential()
  model.add(Embedding(vocab_size, 100, input_length=1))
  model.add(Conv1D(128, 1, activation='relu'))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  print(model.summary())
  # compile network
  model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['accuracy'])
  # fit network
  model.fit(xtrain, ytrain, epochs=10, verbose=2)
  # evaluate
  loss, acc = model.evaluate(xtest, ytest, verbose=0)
  print('Test Accuracy: %f' % (acc*100))


half_text = int(len(preprocessed_text)/2)
half_tags = int(len(tags)/2)

x_train = np.asarray(preprocessed_text[:half_text])
y_train = np.asarray(tags[:half_tags])
x_test = np.asarray(preprocessed_text[half_text+1:])
y_test = np.asarray(tags[half_tags+1:])

cnn_model(x_train, y_train, x_test, y_test, vocab_size)

