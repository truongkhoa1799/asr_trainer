
import os
from signal import signal
import sys
import json
import librosa
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

TRANSFORMED_AUDIO_DIR=Path(sys.argv[1].split('=')[-1])
TEXT_DIR = TRANSFORMED_AUDIO_DIR.joinpath("txt")
AUDIO_DIR = TRANSFORMED_AUDIO_DIR.joinpath("wav")
MANIFESTS_DIR = TRANSFORMED_AUDIO_DIR.joinpath("manifests")

all_manifest_path = MANIFESTS_DIR.joinpath("recorded_manifests.json")
train_manifest_path = MANIFESTS_DIR.joinpath("recorded_train_manifests.json")
test_manifest_path = MANIFESTS_DIR.joinpath("recorded_test_manifests.json")

def create_manifest_item(item):
    try:
        with open(item["sentence_path"], 'r') as fin:
            sentence = fin.readline()
            sentence = sentence.strip()
        
        signal, sr = librosa.load(item["audio_path"].resolve())
        duration = librosa.get_duration(y=signal, sr=sr)
        
        return str(item["audio_path"]), sentence, round(duration, 2)
        
    except Exception as e:
        print(e)
        return None

def save_manifest(manifest_path, list_record):
    with open(manifest_path, mode='a', encoding='utf-8') as fout:
        data = '\n'.join(json.dumps(i, ensure_ascii=False) for i in list_record)
        fout.writelines(data + '\n')
        fout.close()
        
if __name__ == '__main__':
    try:
        data = []
        audios_path = list(AUDIO_DIR.glob("*.wav"))
        
        if not MANIFESTS_DIR.exists():
            MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        if all_manifest_path.exists():
            all_manifest_path.unlink()
        
        def iterator_data(audios_path):
            for audio_path in audios_path:
                file_name = audio_path.name.replace(".wav", ".txt")
                sentence_path = TEXT_DIR.joinpath(file_name)
                
                if not sentence_path.exists() or not audio_path.exists():
                    print(f"Missing audio file {audio_path} or sentence file {sentence_path}")
                    continue
                
                yield {
                    "audio_path": audio_path,
                    "sentence_path": sentence_path
                }
        
        partial_fn = partial(create_manifest_item)
        iterator = iterator_data(audios_path)
        
        p = Pool(6)
        create_manifest_item_map = p.imap_unordered(
            partial_fn,
            tqdm(iterator, total=len(audios_path), desc="[Create manifests]"),
            chunksize=10,
        )
        
        for audio_path, text, duration in create_manifest_item_map:
            manifest_item = {
                "audio_filepath": audio_path,
                "text": text,
                "duration": duration
            }
            
            data.append(manifest_item)
        
        save_manifest(manifest_path=str(all_manifest_path), list_record=data)
        
        train_manifests, test_manifests = train_test_split(data, test_size=0.05, random_state=42)
        save_manifest(manifest_path=str(train_manifest_path), list_record=train_manifests)
        save_manifest(manifest_path=str(test_manifest_path), list_record=test_manifests)
        
        print(f"Number of all manifests: {len(data)}")
        print(f"Number of train manifests: {len(train_manifests)}")
        print(f"Number of test manifests: {len(test_manifests)}")
        exit(0)
    except Exception as e:
        print(e)
        exit(-1)
    