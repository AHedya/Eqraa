from torch.cuda import is_available
from colorama import Fore,Back
from transformers import pipeline 
# import torch
# import yaml
# from trainedModesl import CNNModel,AudioLSTM

device = 'cuda' if is_available else 'cpu' 

class Meta_ASR(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
class SUPER_ASR(metaclass = Meta_ASR):
    _pth= './Models'
    def __init__(self,modelType):
        self.model = self.load_model(modelType)
    def load_model(self,modelType):
        print(Back.GREEN , Fore.YELLOW,
            f'creating ASR {modelType}...' 
            , Back.RESET , Fore.RESET)
        return pipeline('automatic-speech-recognition',f'{SUPER_ASR._pth}/{modelType}' , device=device )
    
    def __call__(self,File):
        return self.model(File)

class ASR_base(SUPER_ASR):
    def __init__(self):
        super().__init__('tarteel-asr')

class ASR_tiny(SUPER_ASR):
    def __init__(self):
        super().__init__('tarteel-asr-tiny')
class ASR_syl(SUPER_ASR):
    def __init__(self):
        super().__init__('syllables')




