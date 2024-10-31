from typing import Annotated
import shutil
import os
import numpy as np

import pyaudio
from Models.models import ASR_base , ASR_syl ,ASR_tiny
from TrainedModels.trainedModesl import KWS_LSTM,KWS_CNN
from torch import no_grad

from fastapi import FastAPI , Query , File, UploadFile , WebSocket

app = FastAPI()



@app.post("/ASR")
async def upload(
        file: Annotated[UploadFile ,File()],
        ASR_type : Annotated[str , Query()] = 'tiny'
        ):
    """revices audio file , writes file down temorary . Performs ASR and deletes file 

    Args:
        file (Annotated[UploadFile ,File): file to apply ASR on
        ASR_type (Annotated[str , Query, optional): determine type of model (base or tiny). Defaults to 'tiny'.

    Returns:
        dictionary : 'message': has transcription of audio  
    """
    try:
        filename = file.filename[:-4]+'server.wav'
        with open(filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            if ASR_type == 'base':
                ASR = ASR_base()
            else:
                ASR = ASR_tiny()
                
            msg=ASR(filename)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        os.remove(filename)
    return {"message": msg['text']}


@app.post("/SYL")
async def upload_SYL(file: Annotated[UploadFile ,File()]):
    """revices audio file , writes file down temorary . Performs syllables and deletes file 

    Args:
        file (Annotated[UploadFile ,File): audio file

    Returns:
        dictionary : 'message': has transcription of audio  
    """
    try:
        filename = file.filename[:-4]+'server.wav'
        with open(filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            ASR = ASR_syl()
            msg=ASR(filename)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        os.remove(filename)
    return {"message": msg['text']}


@app.websocket("/audio_chunks")
async def websocket_audiostreaming_chunks(websocket: WebSocket):
    """recieves stream of audio chunks. performs ASR on each chunk and returns result
    Returns:
        sends back transcription as text response to client side

    """
    await websocket.accept()
    ASR = ASR_base()
    try:
        while True:
            data = await websocket.receive_bytes()
            arr=np.frombuffer(data,dtype=np.float32)
            await websocket.send_text(ASR(arr)['text'])
    except:
        await websocket.close()



from collections import deque
import pyaudio
FORMAT = pyaudio.paFloat32  
CHANNELS = 1  
RATE = 16000  
CHUNK = 1600  
MAX_LEN =30
OVERLAP =20
frames_counter = 0
audio_que = deque(maxlen=MAX_LEN)


import soundfile as sf
@app.websocket("/audio_stream")
async def websocket_audiostreaming(websocket: WebSocket):
    """revices audio stream and perform chuinking to utilize ASR models

    Args:
        depends on global variables
    """
    await websocket.accept()
    frames_counter = 0
    ASR = ASR_base()
    j_counter=0
    try:
        while True:
            data = await websocket.receive_bytes()
            arr_data = np.frombuffer(data , dtype=np.float32)
            audio_que.append(arr_data)
            frames_counter+=1
            if frames_counter % OVERLAP == 0 and len(audio_que)==MAX_LEN:
                frames_counter=0
                wholeChunk=np.r_[*audio_que]
                result = ASR(wholeChunk)
                await websocket.send_text(result['text'])
            
    except:
        await websocket.close()
        


@app.websocket("/broadcast")
async def websocket_audiostreaming(websocket: WebSocket):
    """receives audio stream and streams audio into server's speakers

    Args:
        no args
    """
    await websocket.accept()
    p = pyaudio.PyAudio()
    output_stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True)
    

    try:
        while True:
            data = await websocket.receive_bytes()        
            output_stream.write(data)
    except:
        await websocket.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()




@app.post("/KWS")
async def upload(
        file: Annotated[UploadFile ,File()],
        KWS_type : Annotated[str , Query()] = 'CNN'
        ):
    """Endpoint for uploading audio file to perform KWS (keyword spotting service)

    Args:
        file (Annotated[UploadFile ,File): audio to perform KWS on
        KWS_type (Annotated[str , Query, optional): determine CNN model or LSTM . Defaults to 'CNN'.

    Returns:
        dictionarty : message: list of probability distribution
    """
    try:
        filename = file.filename[:-4]+'server.wav'
        with open(filename, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            if KWS_type == 'CNN':
                KWS = KWS_CNN()
            else:
                KWS = KWS_LSTM()
            
            with no_grad():
                msg=KWS(filename)
            msg= [i.item()for i in msg.squeeze()]
            
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        os.remove(filename)
    return {"message": msg}
