import re
import os
import sys
import librosa
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
from functools import partial
from multiprocess.pool import Pool

MP3_DIR = Path(sys.argv[1].split('=')[-1])
WAV_DIR = Path(sys.argv[2].split('=')[-1])

def processing_mp3(mp3_path, wav_dir):
    wav_path = wav_dir.joinpath(mp3_path.name.replace('.mp3', '.wav'))
    try:
        if wav_path.exists():
            return 0, wav_path
        
        sound = AudioSegment.from_mp3(mp3_path)
        sound = AudioSegment.set_frame_rate(sound, frame_rate=16000)
        sound.export(wav_path, format="wav")
        return 0, wav_path
    except Exception as e:
        return 1, wav_path

if __name__ == '__main__':
    mp3_paths = list(MP3_DIR.glob("*.mp3"))
    if not WAV_DIR.exists():
        WAV_DIR.mkdir(parents=True, exist_ok=True)
    
    def iterator_mp3(mp3_paths):
        for mp3_path in mp3_paths:
            yield mp3_path
            
    p = Pool(24)
    iterator = iterator_mp3(mp3_paths)
    partial_func = partial(processing_mp3, wav_dir=WAV_DIR)
    processing_mp3_map = p.imap_unordered(
        partial_func,
        tqdm(iterator, total=len(mp3_paths), desc="[Processing mp3 audio]")
    )
    
    for code, wav_path in processing_mp3_map:
        if code != 0:
            if wav_path.exists():
                wav_path.unlink()