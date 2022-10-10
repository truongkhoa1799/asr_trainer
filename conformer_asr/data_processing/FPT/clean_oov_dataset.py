import re
import os
import sys
from tabnanny import check
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocess.pool import Pool

from smart_speaker_common.utils.reg_utils import extract_str, replace_str, RegType, chars_to_ignore_regex_begin, chars_to_ignore_regex_end, extract_str

PROCESSED_DATA_DIR = Path(sys.argv[1].split('=')[-1])
text_path = PROCESSED_DATA_DIR.joinpath("noised", "oov_text.txt")
removed_oov_text_path = PROCESSED_DATA_DIR.joinpath("clean", "removed_oov_text.txt")
remained_oov_text_path = PROCESSED_DATA_DIR.joinpath("noised", "remained_oov_text.txt")

chars_to_ignore_regex   = '[\,\?\.\!\;\:\"\'\(\)\{\}\“\‘\”\…]'  # remove special character tokens
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

def normalize_text(item):
    text = re.sub('-n', ' ', item['text'])
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('k\+', 'k cộng', text)
    text = re.sub(chars_to_ignore_regex, ' ', text)
    text = re.sub(" +", " ", text).strip()
    
    is_oov = check_oov(text)
    if is_oov:
        return 1, item['audio_name'], text, item['duration']
    
    return 0, item['audio_name'], text, item['duration']

if __name__ == '__main__':
    ''' Proccess transcriptAll file'''
    clean_texts = []
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
                
               
    iterator = iterator_data(transcript_path=text_path)
    partial_func = partial(normalize_text)
    
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
            oov_texts.append(item)
            
    print(f"Number of clean audio: {len(clean_texts)}")
    print(f"Number of audio has OOV: {len(oov_texts)}")
    
    for data, file_path in zip((clean_texts, oov_texts), (removed_oov_text_path, remained_oov_text_path)):
        with open(file_path, 'w') as fout:
            fout.write("\n".join(data) + "\n")
