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

def get_hps(config_path):
    if config_path==None:
        sys.exit("No config File")
    with open(config_path,'r') as f:
        data=f.read()
    config=json.loads(data)
    hparams=Hparams(**config)
    print(hparams)

    save_path=hparams.train.log_path
    model_name=hparams.train.log_name
    save_path=os.path.join(save_path,model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    hparams.model_dir=save_path

    config_save_path=os.path.join(save_path,"config.json")
    with open(config_save_path,'w') as f:
        f.write(data)

    return hparams

def log_scaler(writer,name, step, loss):
    writer.add_scaler(name, loss, step)

def log_model(writer, audio, model):
    writer.add_graph(model, audio)



class Hparams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = Hparams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()