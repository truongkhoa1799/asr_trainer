import os
import re
import sys
import pickle
import librosa
from tqdm import tqdm
import soundfile as sf
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

AUDIOS_DIR=Path(sys.argv[1].split('=')[-1])
SENTENCE_DICT_PATH=Path(sys.argv[3].split('=')[-1])

TRANSFORMED_AUDIO_DIR=Path(sys.argv[2].split('=')[-1])
TEXT_DIR = TRANSFORMED_AUDIO_DIR.joinpath("txt")
AUDIO_DIR = TRANSFORMED_AUDIO_DIR.joinpath("wav")

chars_to_ignore_regex   = '[\,\?\.\!\;\:\"\'\(\)\{\}\“\‘\”\…]'  # remove special character tokens

def process_audio(src_audio_path, des_audio_path):
    samplerate = 16000
    signal, sr = librosa.load(src_audio_path)
    resample_signal = librosa.resample(y=signal, orig_sr=sr, target_sr=samplerate)
    sf.write(des_audio_path.resolve(), resample_signal, samplerate=samplerate)

def process_text(sentence, des_text_path):
    sentence = re.sub(chars_to_ignore_regex, '', sentence).lower().strip()
    sentence = re.sub(' +', ' ', sentence)
    with open(des_text_path.resolve(), 'w') as fout:
        fout.write(sentence)

if __name__ == "__main__":
    if not TEXT_DIR.exists():
        TEXT_DIR.mkdir(parents=True, exist_ok=True)
    if not AUDIO_DIR.exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not SENTENCE_DICT_PATH.exists():
        print(f"{SENTENCE_DICT_PATH} is not exist")
        exit(-1)

    with open(SENTENCE_DICT_PATH.resolve(), 'rb') as fin:
        sentence_dict = pickle.load(fin)

    try:
        audios_path = list(AUDIOS_DIR.glob("*/*.wav"))
        for audio_path in tqdm(audios_path, total=len(audios_path), desc="Processing approved audios"):
            user = audio_path.parent.name
            
            user_audio_path = AUDIO_DIR.joinpath(f"{user}_{audio_path.name}")
            user_text_path = TEXT_DIR.joinpath(f"{user}_{audio_path.name.replace('.wav', '.txt')}")
            
            if user == "GUEST_USER":
                sentence_id = audio_path.name.split("_")[0]
            else:
                sentence_id, _ = os.path.splitext(audio_path.name)
            sentence = sentence_dict[int(sentence_id.strip())]
            
            process_audio(src_audio_path=audio_path, des_audio_path=user_audio_path)
            process_text(sentence, user_text_path)
            
        exit(0)
    except Exception as e:
        print(e)
        exit(-1)