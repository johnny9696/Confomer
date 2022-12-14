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
from text_cleaner import english_cleaner
import symbols
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
        return (audio, text)

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
        audio=torch.tensor(audio)
        return audio

    def get_lang(self,lang):
        lang=lt.l2num(lang)
        lang=torch.tensor(lang)
        #lang= F.one_hot(lang, num_classes=int(self.hps.model.n_class))
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
        input_length=[x[0].size(1) for x in batch]
        max_target_len=max(input_length)
        input_length=torch.LongTensor(input_length)

        if max_target_len %self.n_frames_per_step!=0:
            assert max_target_len%self.n_frames_per_step==0
        #mel_padding
        mel_padded=torch.FloatTensor(len(batch),n_mels,max_target_len)
        mel_padded.zero_()


        #label tensor set
        label=torch.zeros(len(batch))
        target_length=torch.LongTensor(len(batch))
        target_length.zero_()

        for i in range(0,len(batch)):
            mel=batch[i][0]
            mel_padded[i,:,:mel.size(1)]=mel
            label[i]=batch[i][1]
            target_length[i]=torch.tensor(1, dtype=torch.long)
        label=label.to(torch.long)
        return mel_padded, input_length,  label, target_length

class MelTextLoader(torch.utils.data.Dataset):
    def __init__(self,audiopath_and_lang,hparams):
        self.hps=hparams
        self.audio_lang_list=util.get_text_loader(audiopath_and_lang)
        self.sampling_rate=int(hparams.dataset.sampling_rate)
        self.max_wav_value=float(hparams.dataset.max_wav_value)
        self.add_blank=hparams.dataset.add_blank

        random.seed(1234)
        random.shuffle(self.audio_lang_list)

    def get_mel_text_pair(self,audio_lang_list):
        #data set is based on lj speech
        audio_text_list=audio_lang_list.split('|')
        audio,text=audio_text_list[0],audio_text_list[1]
        audio=self.get_mel(audio)
        text=self.get_text(text)
        return (audio, text)

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
        audio=torch.tensor(audio)
        return audio

    def get_text(self,text):
        text= english_cleaner(text)
        if self.add_blank:
            text=list(text)
            _text=[]
            for i in text:
                _text.append('<BNK>')
                _text.append(i)
            _text.append('<BNK>')
            text=_text
        vec=symbols.text2vec(text)
        vec=torch.tensor(vec)
        return vec

    
    def __getitem__(self,index):
        return self.get_mel_text_pair(self.audio_lang_list[index])
    
    def __len__(self):
        return len(self.audio_lang_list)


class MelTextCollate():
    def __init__(self,n_class, n_frames_per_step=1 ):
        self.n_frames_per_step=n_frames_per_step
        self.n_class=n_class

    def __call__(self, batch):
        """
        Collate Fn : make even length in batch
        mel-spec need to make same size
        Batch :[audio(n_mels,frames),text]
        """
        n_mels=batch[0][0].size(0)
        input_length=[x[0].size(1) for x in batch]
        target_length=[x[1].size(0) for x in batch]
        max_input_len=max(input_length)
        max_target_len=max(target_length)
        input_length=torch.LongTensor(input_length)
        target_length=torch.LongTensor(target_length)

        if max_input_len %self.n_frames_per_step!=0:
            assert max_input_len%self.n_frames_per_step==0
        #mel_padding
        mel_padded=torch.FloatTensor(len(batch),n_mels,max_input_len)
        mel_padded.zero_()

        #label padding
        label_padded=torch.LongTensor(len(batch),max_target_len)
        label_padded.zero_()

        for i in range(0,len(batch)):
            mel=batch[i][0]
            label=batch[i][1]
            mel_padded[i,:,:mel.size(1)]=mel
            label_padded[i,:label.size(0)]=label
            
        label=label.to(torch.long)
        return mel_padded, input_length,  label_padded, target_length
