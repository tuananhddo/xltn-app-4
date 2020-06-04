import os
import threading
import tkinter
import pyaudio
import wave
import librosa
import numpy as np
import os
import math
import joblib

# from GMMHMM import gmmhmm
from sklearn.cluster import KMeans
import hmmlearn.hmm
from time import time
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import log_loss
from scipy.special import softmax
import pickle

TITLE = "Word Reconigtion"
RESOLUTION = "300x150"
BUTTON_CONFIG = {
    'height': 1,
    'width': 15
}
LABEL_CONFIG = {
    'wraplength': 500
}

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
FRAME_PER_BUFFER = 1024

RECORDING_FILE = "temp.wav"
from pydub import AudioSegment

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans


def get_mfcc(file_path):
    y, sr = librosa.load(file_path)  # read .wav file
    hop_length = math.floor(sr * 0.010)  # 10ms hop
    win_length = math.floor(sr * 0.025)  # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1, 1))
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0)  # O^r
    # return T x 36 (transpose of X)
    return X.T  # hmmlearn use T x N matrix


stt = ["tôi", "không", "một", "người", "cáchly"]


class Recorder:
    def __init__(self):
        self.start_button = tkinter.Button(
            root,
            text="Start Recording",
            command=self.start_recording,
            **BUTTON_CONFIG
        )
        self.start_button.pack()
        self.start_lock = False

        self.stop_button = tkinter.Button(
            root,
            text="Stop Recording",
            command=self.stop_recording,
            **BUTTON_CONFIG
        )
        self.stop_button.pack()
        self.stop_lock = True

        self.status = tkinter.Label(
            root,
            text="No recording"
        )
        self.status.pack()

        self.recognize_button = tkinter.Button(
            root,
            text="Recognize Word",
            command=self.recognize,
            **BUTTON_CONFIG
        )
        self.recognize_button.pack()
        self.recognize_lock = True

        self.is_recording = False

    def start_recording(self):
        if self.start_lock:
            return

        self.start_lock = True

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            frames_per_buffer=FRAME_PER_BUFFER,
            input=True
        )

        self.frames = []

        self.is_recording = True
        self.status.config(text="Recording")

        self.recognize_lock = True
        self.stop_lock = False

        thread = threading.Thread(target=self.record)
        thread.start()

    def stop_recording(self):
        if self.stop_lock:
            return

        self.stop_lock = True

        self.is_recording = False

        wave_file = wave.open("temp.wav", "wb")

        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(self.audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)

        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        self.status.config(text="Recorded")

        self.recognize_lock = False
        self.start_lock = False

    def record(self):
        while (self.is_recording):
            data = self.stream.read(FRAME_PER_BUFFER)
            self.frames.append(data)

    def recognize(self):
        dataset = {}
        models = {}
        class_names = ["hoac", "benh_nhan", "nao", "noi", "dich"]
        for cname in class_names:
            file = open('model/' + cname + '.pkl', 'rb')
            m = joblib.load(file)
            models[cname] = m
            file.close()
        # sound = AudioSegment.from_file(RECORDING_FILE, format="wav")
        #
        # start_trim = detect_leading_silence(sound)
        # end_trim = detect_leading_silence(sound.reverse())
        #
        # duration = len(sound)
        # trimmed_sound = sound[start_trim:duration - end_trim]
        # trimmed_sound.export('exx.wav', format='wav')
        # gmmhmm.trim_audio(RECORDING_FILE)
        dataset['model'] = [get_mfcc(RECORDING_FILE)]

        all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)

        kmeans = clustering(all_vectors)
        dataset['model'] = list([kmeans.predict(v).reshape(-1, 1) for v in dataset['model']])
        score = {}
        for O in dataset['model']:
            score = {cname: model.score(O, [len(O)]) for cname, model in models.items() if cname[:4] != 'test'}
        print(score)
        suggestWord = class_names[0];
        maxScore = score[class_names[0]];
        for key,value in score.items():
            if value > maxScore:
                maxScore = value
                suggestWord = key
        print(suggestWord)
        # full_result, result = m.score(dataset
        # nsamples, nx, ny = all_vectors.shape
        # d2_train_dataset = all_vectors.reshape((nsamples, nx * ny))
        # y_predict = [pro_to_label(y_pro) for y_pro in result][0]
        self.status.config(text=f"This is \"{suggestWord}\"")


root = tkinter.Tk()
root.title(TITLE)
root.geometry(RESOLUTION)
app = Recorder()
root.mainloop()
