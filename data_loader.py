import random
import torch
from torch.nn import functional as F
import librosa
from librosa import stft, istft
from librosa.feature import melspectrogram
from librosa.filters import mel
import numpy as np
import torch.utils.data
import util
import lang_type as lt
from scipy.io import wavfile


class MelLangLoader(torch.utils.data.Dataset):
    def __init__(self,audiopath_and_lang,hparams):
        self.hps=hparams
        self.audio_lang_list=util.get_text_loader(audiopath_and_lang)
        self.sampling_rate=int(hparams.dataset.sampling_rate)
        self.max_wav_value=float(hparams.dataset.max_wav_value)

        random.seed(1234)
        random.shuffle(self.audio_lang_list)

    def get_mel_lang_pair(self,audio_lang_list):
        audio_lang_list=audio_lang_list.split('|')
        audio,text=audio_lang_list[0],audio_lang_list[2]
        audio=self.get_mel(audio)
        text=self.get_lang(text)
        return (audio,text)

    def get_mel(self,audio_path):
        sr,audio=wavfile.read(audio_path)
        if sr != self.sampling_rate:
            raise ValueError("{} {} SR doesn`t match target {} SR".format(sr,self.sampling_rate,audio_path))
        audio=audio/self.max_wav_value
        audio= stft(audio,n_fft=int(self.hps.dataset.filter_length),
        hop_length=int(self.hps.dataset.hop_length), win_length=int(self.hps.dataset.win_length),
        window=self.hps.dataset.window)
        mel_filter=mel(sr=sr,n_fft=int(self.hps.dataset.filter_length), fmin=float(self.hps.dataset.f_min),fmax=float(self.hps.dataset.f_max), n_mels=int(self.hps.dataset.n_mels))
        audio=mel_filter.dot(audio)
        """
        audio=melspectorgram(audio, sr=sr,n_fft=int(self.hps.dataset.filter_length),
        hop_length=int(self.hps.dataset.hop_length), win_length=int(self.hps.dataset.win_length),
        window=self.hps.dataset.window,power=float(self.hps.dataset.power))
        """
        audio=torch.Tensor(audio)
        return audio

    def get_lang(self,lang):
        lang=lt.l2num(lang)
        lang=torch.tensor(lang)
        lang= F.one_hot(lang, num_classes=int(self.hps.model.n_class))
        return lang
    
    def __getitem__(self,index):
        return self.get_mel_lang_pair(self.audio_lang_list[index])
    
    def __len__(self):
        return len(self.audio_lang_list)


class MelLangCollate():
    def __init__(self,n_class, n_frames_per_step=1 ):
        self.n_frames_per_step=n_frames_per_step
        self.n_class=n_class

    def __call__(self, batch):
        """
        Collate Fn : make even length in batch
        mel-spec need to make same size
        Batch :[audio(n_mels,frames),text[label]]
        """
        n_mels=batch[0][0].size(0)
        max_target_len=max([x[0].size(1) for x in batch])
        if max_target_len %self.n_frames_per_step!=0:
            assert max_target_len%self.n_frames_per_step==0
        #mel_padding
        mel_padded=torch.FloatTensor(len(batch),n_mels,max_target_len)
        mel_padded.zero_()

        #label tensor set
        label=torch.zeros(len(batch),self.n_class)

        for i in range(len(batch[0])):
            mel=batch[i][0]
            mel_padded[i,:,:mel.size(1)]=mel
            label[i]=batch[i][1]
        label=label.to(torch.long)
        return mel_padded, label
