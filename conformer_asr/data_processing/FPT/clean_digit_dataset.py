import re
import os
import sys
from tabnanny import check
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocess.pool import Pool
from conformer_asr.utils import Logger

from conformer_asr.reg_utils import extract_str, replace_str, RegType, chars_to_ignore_regex_begin, chars_to_ignore_regex_end, extract_str

LOGGER = Logger(name="CLEANNG_DIGIT_FPT_DATASET")

PROCESSED_DATA_DIR = Path(sys.argv[1].split('=')[-1])
text_path = PROCESSED_DATA_DIR.joinpath("noised", "number_text.txt")
removed_number_text_path = PROCESSED_DATA_DIR.joinpath("clean", "removed_number_text.txt")
remained_number_text_path = PROCESSED_DATA_DIR.joinpath("noised", "remained_number_text.txt")

chars_to_ignore_regex   = '[\,\?\.\!\;\:\"\'\(\)\{\}\“\‘\”\…]'
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
    '''
    Preprocessing data pipeline
        1: extract date
        2: extract time
        3: number
        
    '''
    data = re.sub(chars_to_ignore_regex_begin, ' ', item['text']).lower().strip()
    
    list_match = extract_str(RegType.NUMBER, text = data)
    data = replace_str(RegType.NUMBER, list_match, data)
    
    list_match = extract_str(RegType.CURRENCY, text = data)
    data = replace_str(RegType.CURRENCY, list_match, data)
        
    data = re.sub(chars_to_ignore_regex_end, ' ', data)
    data = re.sub(" +", " ", data).strip().lower()
    
    has_digit = re.search('[0-9]', data)
    is_oov = check_oov(data)
    if has_digit is not None or is_oov:
        return 1, item['audio_name'], data, item['duration']
    
    return 0, item['audio_name'], data, item['duration']

if __name__ == '__main__':
    try:
        ''' Proccess transcriptAll file'''
        clean_texts = []
        digit_texts = []
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
                digit_texts.append(item)
                
        LOGGER.log_info(f"Number of clean audio: {len(clean_texts)}")
        LOGGER.log_info(f"Number of audio has digit: {len(digit_texts)}")
        
        for data, file_path in zip((clean_texts, digit_texts), (removed_number_text_path, remained_number_text_path)):
            with open(file_path, 'w') as fout:
                fout.write("\n".join(data) + "\n")
    except Exception as e:
        print(e)
        exit(-1)