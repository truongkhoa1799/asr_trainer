from genericpath import exists
import re
import os
import sys
from tabnanny import check
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocess.pool import Pool

'''
This dataset has some special character such as:
    - \r\n
    - -N
'''

TRANSCRIPT_PATH = Path(sys.argv[1].split('=')[-1])
PROCESSED_DATA_DIR = Path(sys.argv[2].split('=')[-1])

clean_text_path = PROCESSED_DATA_DIR.joinpath("clean", "clean_text.txt")
oov_text_path = PROCESSED_DATA_DIR.joinpath("noised", "oov_text.txt") 
digit_text_path = PROCESSED_DATA_DIR.joinpath("noised", "number_text.txt")

vocabs = [
    'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x',

    'a', 'á', 'à', 'ạ', 'ã', 'ả',
    'ă', 'ắ', 'ằ', 'ặ', 'ẵ', 'ẳ',
    'â', 'ấ', 'ầ', 'ậ', 'ẫ', 'ẩ',

    'e', 'é', 'è', 'ẹ', 'ẽ', 'ẻ',
    'ê', 'ế', 'ề', 'ệ', 'ễ', 'ể',

    'i', 'í', 'ì', 'ị', 'ĩ', 'ỉ',
    'y', 'ý', 'ỳ', 'ỵ', 'ỹ', 'ỷ',
    
    'o', 'ó', 'ò', 'ọ', 'õ', 'ỏ',
    'ô', 'ố', 'ồ', 'ộ', 'ỗ', 'ổ',
    'ơ', 'ớ', 'ờ', 'ợ', 'ỡ', 'ở',

    'u', 'ú', 'ù', 'ụ', 'ũ', 'ủ',
    'ư', 'ứ', 'ừ', 'ự', 'ữ', 'ử',
    
    'j', 'f', 'w', 'z', ' '
]

def check_oov(text):
    for i in text:
        if i not in vocabs:
            return True
    return False

def process_item(item):
    is_oov = check_oov(item['text'])
    if not is_oov:
        return 0, item['audio_name'], item['text'], item['duration']

    has_digit = re.search('[0-9]', item['text'])
    if has_digit is not None:
        return 1, item['audio_name'], item['text'], item['duration']
    
    return 2, item['audio_name'], item['text'], item['duration']

if __name__ == '__main__':
    try:
        ''' Proccess transcriptAll file'''
        clean_texts = []
        digit_texts = []
        oov_texts = []
        def iterator_data(transcript_path):
            with open(transcript_path, 'r') as fin:
                for transcript in fin:
                    transcript_splitted = transcript.split('|')
                    audio_name = transcript_splitted[0]
                    text = transcript_splitted[1]
                    duration = round(float(transcript_splitted[2].split('-')[-1]), 2)
                    text = text.strip().lower()
                    
                    yield {
                        "audio_name": audio_name,
                        "text": text,
                        "duration": duration,
                    }
                
        iterator = iterator_data(transcript_path=TRANSCRIPT_PATH)
        partial_func = partial(process_item)
        
        p = Pool(8)
        process_item_map = p.imap_unordered(
            partial_func,
            tqdm(iterator, desc="Process items FPT")
        )
        
        for code, audio_name, text, duration in process_item_map:
            item = "|".join([audio_name, text, str(duration)])
            if code == 0:
                clean_texts.append(item)
            elif code == 1:
                digit_texts.append(item)
            elif code == 2:
                oov_texts.append(item)
        
        print(f"Number of clean audio: {len(clean_texts)}")
        print(f"Number of audio has digit: {len(digit_texts)}")
        print(f"Number of audio oov: {len(oov_texts)}")
        for data, file_path in zip((clean_texts, digit_texts, oov_texts), (clean_text_path, digit_text_path, oov_text_path)):
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
            with open(file_path, 'w') as fout:
                fout.write("\n".join(data) + "\n")
    except Exception as e:
        print(e)
        exit(-1)
