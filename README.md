# Eqraa

Eqraa is a project launched to aid in the development of recitation and memorization of the nobel Quran. This project leverages machine learning techniques imporves pronunciation, and provides feedback for better interactive experience. 

## Features

- **Speech Recognition:** Analyzes spoken words to provide feedback and ensure accurate pronunciation.
- **Keyword Spotting System:** Check how close is the spoken word to the optimal pronunciation. 
- **Interactive Reading:** Provides real-time assistance and feedback during reading.


## Installation

To install and run Eqraa locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AHedya/Eqraa.git
   cd Eqraa
2. **Initialize Anaconca env:**
   ```sh
   conda env create -f environment.yml
   conda activate torch-gpu

   // or whatever name you choose for the environment
3. **Run the server:**
   ```bash
   fastapi dev server.py
  make sure server running at (http://127.0.0.1:8000)

4. **Try client side:**
   Open `client.ipynb`, and now you can try the project yourself.


## Demo 

  You can find demo of the project running at [project demo](https://drive.google.com/file/d/1abUFtnbbhiRrzCDPMjIxM4u5JRm_Tc5O/view?usp=sharing).
  
  Training results, and additional details at [implemntation docs](https://docs.google.com/document/d/12FbvVgNvbhm59A_mEbM1HwJ_G_V0PDv_tZrm-wCAPXk/edit?usp=sharing), page 10 to the end 
  
  All scripts and trained models variations are at [scripts](https://drive.google.com/file/d/1NK2WxmFg4H_4FoxT24VAOG8fDvHLjZ6a/view?usp=sharing)


## Notes
   - To accomplish this project we used our from-scratch deep learning models, packaged the into singleton classes, And also we used some pretrained models such as [tarteel-base](https://huggingface.co/tarteel-ai/whisper-base-ar-quran),[tarteel-tiny](https://huggingface.co/tarteel-ai/whisper-tiny-ar-quran) and [Quran syllables](https://huggingface.co/IbrahimSalah/Wav2vecXXl_quran_syllables)

   - Pre-trained models are ready to be utilized on the whole Moshaf, although we only used them on AlFatiha, while our from-scratch models can only be applied on AlFatiha.

## Data set

  Second version of our data set is available on [Kaggle](https://www.kaggle.com/datasets/ahedya/labeled-quran-dataset). Make sure You're viewing second version.



   


   
