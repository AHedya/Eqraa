
from torch.cuda import is_available
from colorama import Fore,Back
from transformers import pipeline 
import torch
import yaml
from .architectures import CNNModel , AudioLSTM



device = 'cuda' if is_available else 'cpu' 


class Meta_KWS(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class KWS_CNN(metaclass=Meta_KWS):
    def __init__(self):
        self.model = CNNModel().to(device)
        print(Back.GREEN , Fore.YELLOW,
            f'creating KWS_CNN...' 
            , Back.RESET , Fore.RESET)
        self.model.load_state_dict(torch.load(r"./TrainedModels/models/CNN_fineTuned_25_mixup.pt"))
        self.model.eval()
    
    def __call__(self,filePath):
        self._preprocess(filePath)
        return self.model(self.signal.unsqueeze(0).unsqueeze(0))
    
    def _preprocess(self,filePath):
        if type(filePath) == str:
            signal , sr = librosa.load(filePath,sr=16_000)
        signal = resample_if_necessary(signal,sr)
        #signal = silence(signal)
        signal = pad_trunc_if_necessary(signal)
        signal = SK_Normalize_signal(signal)
        signal = transformation(signal)
        self.signal = torch.from_numpy(signal).to(torch.float32).to(device)
        


class KWS_LSTM(metaclass=Meta_KWS):
    
    def __init__(self,db=20):
        self.db=db
        
        self.model = AudioLSTM().to(device)
        print(Back.GREEN , Fore.YELLOW,
            f'creating KWS_LSTM...' 
            , Back.RESET , Fore.RESET)
        self.model.load_state_dict(torch.load(r"./TrainedModels/models/LSTM_150_state_kaggle.pt"))
        self.model.eval()
    def __call__(self,filePath):
        self._preprocess(filePath)
        return self.model(self.signal.unsqueeze(0))
    
    def _preprocess(self,filePath):
        if type(filePath) == str:
            signal , sr = librosa.load(filePath,sr=16_000)
        signal = resample_if_necessary(signal,sr)
        signal = silence(signal ,db = self.db)
        #signal = pad_trunc_if_necessary(signal)
        signal = SK_Normalize_signal(signal)
        signal = transformation(signal)
        self.signal = torch.from_numpy(signal.T).to(torch.float32).to(device)
        
    def adjust_db(self,db):
        self.db=db



import librosa
from sklearn.preprocessing import minmax_scale
import numpy as np
from dataclasses import dataclass

with open('./TrainedModels/models/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


@dataclass
class Params:
    NUM_MELS: int
    SAMPLES_TO_CONSIDER: int
    TARGET_SR: int
    hop_length:int
    n_fft:int
    
p =  Params(**config['features'])
    
def pad_trunc_if_necessary(signal):
        if len(signal) > p.SAMPLES_TO_CONSIDER:
            signal = signal[:p.SAMPLES_TO_CONSIDER]
        elif len(signal) < p.SAMPLES_TO_CONSIDER:
            zeros_to_pad = p.SAMPLES_TO_CONSIDER - len(signal)
            pad_first = zeros_to_pad//2
            zeros_to_pad -=pad_first
            signal = np.r_[np.zeros(int(pad_first)),signal,np.zeros(int(zeros_to_pad))]
            
        return signal
    
def SK_Normalize_signal(signal , axis = 0):
    return minmax_scale(signal,axis=axis)


def resample_if_necessary(signal,sr):
    if sr != p.TARGET_SR :
        signal = librosa.resample(y=signal,orig_sr=sr,target_sr=p.TARGET_SR)
    return signal

def transformation(signal):
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=p.TARGET_SR,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        n_mels=p.NUM_MELS
    )
    mel = librosa.power_to_db(mel)
    return mel

def silence(signal,db=20):
    intervals = librosa.effects.split(signal,top_db=db)
    aggregated_audio = np.empty(0)
    for start, end in intervals:
        segment = signal[start:end]
        aggregated_audio = np.concatenate((aggregated_audio, segment))

    return  aggregated_audio
def tamble(signal):
    signal[:500] *=np.linspace(0,1,500)
    signal[-500:] *=np.linspace(1,0,500)
    return signal


