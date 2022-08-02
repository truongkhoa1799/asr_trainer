import os
import json
import soundfile as sf
import numpy as np
import glob

from joblib import Parallel, delayed
from pydub import AudioSegment
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split

transcript_path = "/home/nhan/NovaIntechs/data/ASR_Data/FPT/transcriptAll.txt"
FPT_mp3_dir = "/home/nhan/NovaIntechs/data/ASR_Data/FPT/mp3"
FPT_wav_dir = "/home/nhan/NovaIntechs/data/ASR_Data/FPT/wav"

FPT_manifest_dir = "/home/nhan/NovaIntechs/data/ASR_Data/FPT/manifests"
FPT_test_manifest = os.path.join(FPT_manifest_dir, "FPT_test_manifest.json")
FPT_train_manifest = os.path.join(FPT_manifest_dir, "FPT_train_manifest.json")

transcript_dict = dict()
list_row = list()


def convert_function(file_path, des):
    head, filename = os.path.split(file_path)
    filename, ext = os.path.splitext(filename)
    new_filename = filename + '.wav'
    new_wav_path = os.path.join(des, new_filename)
    if os.path.exists(new_wav_path): return None

    sound = AudioSegment.from_mp3(file_path)
    sound = AudioSegment.set_frame_rate(sound, frame_rate=16000)
    sound.export(new_wav_path, format="wav")
    return None

def validate_data(file_path):
    _, sr = sf.read(file_path)
    if sr != 16000 or not str(file_path).endswith(".wav"): return file_path
    else: return "0"


def convert_mp3_wav(src, des):
    print("Start convert mp3 to wav")
    mp3_files = glob.glob(os.path.join(src, "*.mp3"))
    if not os.path.exists(des):
        os.makedirs(des)

    num_wavs = len(os.listdir(des))
    print("Nums of files in mp3: {}.".format(len(mp3_files)))
    if num_wavs == len(mp3_files):
        print("This mp3 file is already converted to wav")
    else:
        with Parallel(n_jobs=8, verbose=10) as parallel:
            list_invalid = parallel(
                delayed(convert_function)(mp3_file, des) for mp3_file in mp3_files
            )

    num_wavs = len(os.listdir(des))
    print("Finish convert mp3 to wav with nums of files {}".format(num_wavs))
    print()


    print("Start validate sample rate and format")
    wav_files = glob.glob(os.path.join(des, "*.wav"))
    with Parallel(n_jobs=8, verbose=10) as parallel:
        list_invalid = parallel(
            delayed(validate_data)(wav_file) for wav_file in wav_files
        )
    print("Finish validate sample rate and format with invalid files: {}".format(list_invalid.count(list_invalid != "0")))

def create_FPT_manifest():
    with open(transcript_path, mode='r') as fin:
        transcript = fin.readlines()
        for i in transcript:
            row = i.split("|")
            if len(row) != 3: continue
            filename, ext = os.path.splitext(row[0])
            new_filename = filename + '.wav'
            transcript_dict[new_filename] = {
                "text": row[1].strip().lower(),
                "duration": float(row[2].split("-")[-1])
            }

    print("Number of file transcript row: {}".format(len(transcript_dict.keys())))
    for wav in os.listdir(FPT_wav_dir):
        text = transcript_dict[wav]["text"]
        duration = transcript_dict[wav]["duration"]
        wav_path = os.path.join(FPT_wav_dir, wav)
        data = {'audio_filepath': wav_path, "duration": duration, 'text': text}
        list_row.append(data)

    train_rows, test_rows = train_test_split(list_row, train_size=TRAIN_PER, test_size=TEST_PER, random_state=42)
    print("Number of data train: {}".format(len(train_rows)))
    print("Number of data test: {}".format(len(test_rows)))

    with open(FPT_test_manifest, mode='w', encoding='utf-8') as f:
        for i in test_rows:
            f.write(
                json.dumps(i, ensure_ascii=False) + '\n'
            )
    with open(FPT_train_manifest, mode='w', encoding='utf-8') as f:
        for i in train_rows:
            f.write(
                json.dumps(i, ensure_ascii=False) + '\n'
            )
            
# convert_mp3_wav(FPT_mp3_dir, FPT_wav_dir)
# create_FPT_manifest()