import librosa
import numpy as np
import sys
import os
import json
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pylab as plt


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

def label2length_label(label, output_length, hps):
  result=[]
  for i in range(0,int(hps.train.batch_size)):
    temp=[label[i] for x in range(0,output_length)]
    result.append(temp)
  return torch.tensor(result)

def RGBA2RGB(RGBA):
  row,col,ch =RGBA.shape
  if ch ==3:
    return RGBA
  assert ch==4

  rgb=np.zeros((row,col,3),dtype='float32')
  r,g,b,a=RGBA[:,:,0],RGBA[:,:,1],RGBA[:,:,2],RGBA[:,:,3]
  a= np.asarray(a,dtype='float32')/255.0
  R,G,B=(255,255,255)

  rgb[:,:,0]=r*a+(1.0-a)*R
  rgb[:,:,1]=g*a+(1.0-a)*G
  rgb[:,:,2]=b*a+(1.0-a)*B

  return np.asarray(rgb, dtype='uint8')


def plot_atten_to_numpy(label, target, atten):
  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(atten, aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  
  xlabel = 'Decoder timestep'
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  fig.canvas.draw()
  data=np.array(fig.canvas.renderer._renderer)
  #print(data.shape)
  #data=RGBA2RGB(data)
  plt.close()
  return data


def logger_start(hps):
  comment=hps.train.log_name+"_lr_"+str(hps.train.learning_rate)+"_batch_size_"+str(hps.train.batch_size)
  log_path = os.path.join(hps.train.log_path,comment)
  writer=SummaryWriter(log_dir=log_path)
  return writer

def log_scalar(writer,name, step, loss):
  writer.add_scalar(name, loss, step)

def log_model(writer, audio, model):
  writer.add_graph(model, audio)

def log_atten(writer, name, step, image):
  writer.add_image(name, image, step,dataformats='HWC')
  
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