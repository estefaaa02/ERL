#Install SpeechRecognition library, which has support for several engines and APIs, online and offline.
#pip install SpeechRecognition
#Install dependencies
#!apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
#Install pyAudio library
#pip install pyAudio
#pip install --upgrade google-cloud-speech

#Import libaries for the speech to text and text preprocessing
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd

#Define a variable and assign an instance of the recognizer class to it
r = sr.Recognizer()

"""
This section takes a zip file which contains the corpus and extracts all the audio files from it
"""
from zipfile import ZipFile

# specifying the zip file name
file_name = "corpus.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
#The file is passed to the class AudioFile, which does a necessary preprocessing step to avoid errors related to the file's data type
harvard = sr.AudioFile('f_ans001aes.wav')
with harvard as source:
  #The record method is called to convert the audio file to an audio data type
  audio = r.record(source)
  #Uses Google's free web search API to do the speech to text
  texto = r.recognize_google(audio, language="es-CO")
  print(texto)

#Opens the file that contains the names of the audio files
f = open("corpus.txt", "r")
#Creates arrays to persist the file names, text of the audio and the emotion of said audio
nombresArchivos = []
contenidosArchivos = []
emociones = []
#Iterates through the lines of the corpus.txt file to get the names of the audio files
for linea in f:
  #If the line contains a line break, it is removed. Otherwise, the whole line is passed to the AudioFile class
  if linea[-1]== "\n":
    harvard = sr.AudioFile(linea[:-1])
  else:
    harvard = sr.AudioFile(linea)

  with harvard as source:
    #The record method is called to convert the audio file to an audio data type
    audio = r.record(source)
    #The try/except block will catch any errors given by the recognize_google method
    try:
      #Uses Google's free web search API to do the speech to text
      texto = r.recognize_google(audio, language="es-CO")
    except:
      #If an error is given by the method, then the file is ignored and the loop continues to the next
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
print(df)

nltk.download('punkt')
tokens = []
#Iterates through the texts obtained from the files
for sentence in contenidosArchivos:
  #The sentence of the audio file is normalized using the method created
  normalized = normalize(sentence)
  #The normalized sentence is tokenized and saved in an array
  tokens.append(nltk.word_tokenize(normalized))

print(tokens)

nltk.download('stopwords')
#The spanish stop words are assigned to a variable for further use
stop_words = set(stopwords.words('spanish'))
print(stop_words)
tokensNoStopWords = []

#Iterates through the previously obtained tokens
for sentenceTokens in tokens:
  filtered_sentence = [w for w in sentenceTokens if not w in stop_words]
  filtered_sentence = []
  #Filters through the tokens and the stop words are taken out
  for w in sentenceTokens:
    if w not in stop_words:
        filtered_sentence.append(w)
  tokensNoStopWords.append(filtered_sentence)
print(tokensNoStopWords)

nltk.download('wordnet')
stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()
stemmed_words = []
#Iterates through the tokens without stop words
for words in tokensNoStopWords:
  for word in words:
    #Each word is stemmed and lemmatized
    stemmed_word = stemmer.stem(word)
    stemmed_words.append(lemmatizer.lemmatize(stemmed_word))

print(stemmed_words)

nltk.download('averaged_perceptron_tagger')
tagged = []
#The stemmed and lemmatized words are then tagged
tagged.append(nltk.pos_tag(stemmed_words))

print(tagged)

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

def pottsAlgotithm(textString):
  sentences = textString.split('.')
  for index, sentence in enumerate(sentences):
    sentences[index] = pottsBySentence(sentence)
  print(sentences)

pottsAlgotithm('Ayer no tuve un buen día')
