import librosa
import numpy
import sys
import os

def get_wav(audio_path):
    wav,sr=librosa.load(audio_path)
    return wav, sr