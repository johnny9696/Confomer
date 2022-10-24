import random
import torch
import liborsa
from librosa import stft, istft
import librosa.feature.melspectrogram as melspectorgram
import numpy as np
import torch.util.data
import util
import lang_type as lt


class MelLangLoader(torch.utils.data.Dataset):
    def __init__(self,audiopath_and_lang,hparams):
        self.hps=hparams
        self.audio_lang_list=util.get_text_loader(audiopath_and_lang)
        self.sampling_rate=hparams.dataset.sampling_rate
        self.max_wav_value=hparams.dataset.max_wav_value

        random.seed(1234)
        random.shuffile(self.audio_lang_list)

    def get_mel_lang_pair(self,audio_lang_list):
        audio,text=audio_lang_list[0],audio_lang_list[2]
        audio=self.get_mel(audio)
        text=self.get_text(text)
        return (audio,text)

    def get_mel(self,audio_path):
        audio,sr=librosa.load(audio_path)
        if sr != self.sampling_rate:
            raise ValueError("{} {} SR doesn`t match target {} SR".format(sr,self.sampling_rate,audio_path))
        audio_norm=audio/self.max_wav_value
        audio= stft(audio,n_fft=self.hps.dataset.filter_length,
        hop_length=self.hps.dataset.hop_length, win_length=self.hps.dataset.win_length,
        window=self.hps.dataset.window)
        audio=melspectorgram(audio, sr=sr,n_fft=self.hps.dataset.filter_length,
        hop_length=self.hps.dataset.hop_length, win_length=self.hps.dataset.win_length,
        window=self.hps.dataset.window,power=self.hps.dataset.power_)
        audio=torch.Tensor(audio)
        return audio

    def get_lang(self,lang):
        lang=lt.l2num(lang)
        lang=torch.IntTensor(lang)
        return lang
    
    def __getitem(self,index):
        return self.get_mel_lang_pair(self.audio_lang_list[index])
    
    def __len__(self):
        return len(self.audio_lang_list)


class MelLangCollate():
    def __init__(self,n_frames_per_step=1):
        self.n_frames_per_step=n_frames_per_step

    def __call__(self, batch):
        """
        Collate Fn : make even length in batch
        mel-spec need to make same size
        Batch :[audio(n_mels,frames),text[label]]
        """
        n_mels=batch[0][0].size(0)
        max_target_len=max([x[1].size(1) for x in batch])
        if max_target_len %self.n_frames_per_step!=0:
            assert max_target_len%self.n_frames_per_step==0
        #mel_padding
        mel_padded=torch.FloatTensor(len(batch),n_mels,max_target_len)
        mel_padded.zero()

        #label tensor set
        label=torch.IntTensor(len(batch))

        for i in range(batch.size(0)):
            mel=batch[i][0]
            mel_padded[i,:,:mel.size(1)]=mel
            label[i]=batch[i][1]

        return mel_padded, label
