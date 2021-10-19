# ERL: Emotion Recognition Library
ERL is a Python library that does emotion recognition through audio. With this library you can recognize emotions using audio signals, the text extracted from the audio or using both.

## Installation
 - Clone the source of this library: `https://github.com/estefaaa02/ERL.git`
 - Install library: `pip install -e .`
### Note:
- If using Windows download the PyAudio wheel for Windows and install it using `pip install`.
- If using Linux install the PyAudio library using `pip install pyaudio`
- If using Linux run `apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg`
- By doing `pip install -e . `you are installing the necessary dependencies for the library to work
## Emotion classification example

    from ERL import PredictionsModule
    
    print("Predicción Audio: ", PredictionsModule.predict_emotion_audio_svm('ERL/data/es/f_ans002aes.wav'))
    print("Predicción Texto: ", PredictionsModule.predict_emotion_text_cnn('ERL/data/es/f_ans002aes.wav'))
    print("Predicción Bimodal: ", PredictionsModule.predict_emotion_bimodal('ERL/data/es/f_ans002aes.wav'))
