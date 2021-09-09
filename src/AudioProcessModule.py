#The pyAudioAnalysis library is installed for the preprocessing of the audio signals
#Estas librerías deben ser instaladas, necesario validar.
#pip install pyAudioAnalysis
#pip install AudioSegment
#pip install eyed3
#------------------------------------------

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
import IPython
import numpy as np
import plotly.graph_objs as go
import plotly
import os

audio_dir = os.path.dirname(__file__)

#The read_audio_file method returns the sampling rate of the audio file and a Numpy array of the audio samples
[Fs, x] = audioBasicIO.read_audio_file(audio_dir + "\\wav_corpus\\f_ans001aes.wav")
#The feature_extraction function returns (a) 68x20 short-term feature matrix, where 68 is the number of short-term features implemented in the library and
#20 is the number of frames that fit into the 1-sec segments and (b) a 68-length list of strings that contain the names of each feature implemented in the library.
F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
#The audio files are represented in a chart
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

# read audio data from file
# (returns sampling freq and signal as a numpy array)
fs, s = audioBasicIO.read_audio_file(audio_dir + "\\wav_corpus\\f_ans001aes.wav")
# play the initial and the generated files in notebook:
IPython.display.display(IPython.display.Audio(audio_dir + "\\wav_corpus\\f_ans001aes.wav"))
# print duration in seconds:
duration = len(s) / float(fs)
print(f'duration = {duration} seconds')
# extract short-term features using a 50msec non-overlapping windows
win, step = 0.050, 0.050
[f, fn] = ShortTermFeatures.feature_extraction(s, fs, int(fs * win),
                                int(fs * step))
print(f'{f.shape[1]} frames, {f.shape[0]} short-term features')
print('Feature names:')
for i, nam in enumerate(fn):
    print(f'{i}:{nam}')
# plot short-term energy
# create time axis in seconds
time = np.arange(0, duration - step, win)
# get the feature whose name is 'energy'
energy = f[fn.index('energy'), :]
mylayout = go.Layout(yaxis=dict(title="frame energy value"),
                     xaxis=dict(title="time (sec)"))
plotly.offline.iplot(go.Figure(data=[go.Scatter(x=time,
                                                y=energy)],
                               layout=mylayout))