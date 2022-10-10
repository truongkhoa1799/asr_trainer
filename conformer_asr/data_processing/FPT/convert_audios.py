import re
import os
import sys
from tabnanny import check
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocess.pool import Pool

MP3_DIR = Path(sys.argv[1].split('=')[-1])
WAV_DIR = Path(sys.argv[2].split('=')[-1])

if __name__ == '__main__':
    mp3_paths = MP3_DIR.glob("*.mp3")
    print(mp3_paths[0])