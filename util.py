import librosa
import numpy
import sys
import os
import json


def get_text_loader(text_path):
    with open(text_path,'r',encoding='utf-8') as fi:
        data=fi.read().split('\n')
    return data

def get_wav(audio_path):
    wav,sr=librosa.load(audio_path)
    return wav, sr

def get_hps():