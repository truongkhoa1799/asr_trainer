import os
import csv
import copy
import json
from pathlib import Path
import librosa
from typing import AnyStr, List, TypedDict
import joblib
from tqdm.auto import tqdm

from functools import partial
from multiprocessing import Pool


import numpy as np
import soundfile as sf
import argparse

import torch
import kenlm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

from jiwer import wer
from conformer_asr.utils import Logger, read_manifest, config

# Init ASR model
processor = None
model = None
ngram_lm_model = None
LOGGER = Logger(name="EVALUATING_ASR_DATA")

class EvaluationLog(TypedDict):
    audio_filepath: str
    ground_truth_text: str
    transcribed_text: str
    wer: int = 0

class EvaluationConfig(TypedDict):
    original_manifest_path: str
    log_dir: str
    dataset_name: str
    device: str

class CleanConfig(TypedDict):
    manifest_path: str
    original_manifest_path: str
    log_dir: str
    dataset_name: str
    threshold_wer: float

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    vocab_list[tokenizer.pad_token_id] = ""
    vocab_list[tokenizer.unk_token_id] = ""
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    return decoder


def transcribe_ASR(raw_signal, device='cpu'):
    """Transbribe audio signal to text
    
    Args:
        raw_signal (ndarray): Audio loaded from librosa.load()

    Returns:
        AnyStr: Text of audio signal
    """

    signal = np.asarray(raw_signal, dtype=np.float32).flatten()
    input_values = processor(
        signal,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values.to(device)
    logits = model(input_values).logits[0]
    beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=200)
    return beam_search_output

def save_log_result(log_data, log_data_result_path):
    fieldnames = ['audio_filepath', 'ground_truth_text', 'transcribed_text', 'wer']
    datas = list(map(lambda log: list(log.values()), log_data))

    is_files_exist = os.path.exists(str(log_data_result_path))
    f = open(str(log_data_result_path), 'a', encoding='UTF8', newline='')
    
    writer = csv.writer(f)
    if not is_files_exist:
        writer.writerow(fieldnames)
    writer.writerows(datas)
    f.close()

def evaluate_data(config):
    '''
    Evaluating data for dataset
    Args:
        dataset_name: dataset name of evaluation dataset
        evaluatation_manifest_path: manifest_path of evaluation dataset
        log_directory: log_directory for storing result. Error and Resuls
    Results:
        Save result in format: [audio_filepath,ground_truth_text,transcribed_text,wer]
        Save audio_path of error_files
    '''
    dataset_name = config['dataset_name']
    evaluatation_manifest_path = Path(config['original_manifest_path'])
    log_directory = Path(config['log_dir']).joinpath(dataset_name)
    
    LOGGER.log_info(f"Evaluating data from dataset {dataset_name}")
    
    if not log_directory.exists():
        log_directory.mkdir(parents=True, exist_ok=True)
    log_data_result_path = log_directory.joinpath(f"{dataset_name}_result.log")
    log_data_error_path = log_directory.joinpath(f"{dataset_name}_error.log")
    
    # Clear old log results
    if log_data_result_path.exists(): 
        log_data_result_path.unlink(missing_ok=True)
    if log_data_error_path.exists(): 
        log_data_error_path.unlink(missing_ok=True)
    
    # Inititalize parameters
    counts = 0
    log_data = []
    error_files = list()
        
    manifest_data, _ = read_manifest(str(evaluatation_manifest_path))
    for data in tqdm(manifest_data, total=len(manifest_data), desc=f"Evaluating {dataset_name}"):
        try:
            signal, _ = librosa.load(data["audio_filepath"], sr=16000)
            transcribed_text = transcribe_ASR(signal, device=config['device'])
            ground_truth_text = data["text"]
            wer_score = wer([transcribed_text], [ground_truth_text])
            
            evaluation_log = EvaluationLog(
                audio_filepath=data["audio_filepath"],
                ground_truth_text=ground_truth_text,
                transcribed_text=transcribed_text,
                wer = wer_score
            )
            log_data.append(evaluation_log)
            
        except Exception as e:
            print(e)
            error_files.append(data["audio_filepath"])
        
        counts += 1
        if counts % 500 == 0:
            save_log_result(log_data, log_data_result_path)
            log_data.clear()
        
        # if counts >= 14: break

    if len(log_data) != 0:
        LOGGER.log_info("Save last data logs")
        save_log_result(log_data, log_data_result_path)
        log_data = []
    
    # Save error file
    with open(log_data_error_path, mode='w', encoding='utf-8') as fout:
        error_result = "\n".join(error_files)
        fout.writelines(error_result + '\n')
        fout.close()


def validate_evaluation_results(row, threshold):
    '''
    Validate evaluation results
    Args: row
    Results:
        code:
            - 1: rejected
            - 0: accept
        data: audio_path
        
    '''
    row = row.replace("\n", "")
    row_values = row.split(",")
    audio_path = row_values[0]
    wer = row_values[-1]
    
    if float(wer) > float(threshold):
        return 1, audio_path
    
    return 0, audio_path

def extract_error_results(audio_path):
    audio_path = audio_path.replace("\n", "")
    return audio_path
    
def clean_bad_data(config):
    '''
    Clean error and audio with WER > threshold
    '''
    dataset_name = config['dataset_name']
    manifest_path = Path(config['manifest_path'])
    original_manifest_path = Path(config['original_manifest_path'])
    log_directory = Path(config['log_dir']).joinpath(dataset_name)
    
    LOGGER.log_info(f"Cleaning data from dataset {dataset_name}")
    log_data_result_path = log_directory.joinpath(f"{dataset_name}_result.log")
    log_data_error_path = log_directory.joinpath(f"{dataset_name}_error.log")
    
    if not log_data_result_path.exists() or not log_data_error_path.exists():
        LOGGER.log_error("Invalid log_data_result_path or log_data_error_path")
        return
    
    if not original_manifest_path.exists():
        LOGGER.log_error("Invalid original_manifest_path")
        return
    
    if manifest_path.exists():
        manifest_path.unlink(missing_ok=True)
        
    def iterator(log_data_result_path, skip_first=False):
        with open(log_data_result_path, 'r') as fin:
            for idx, row in enumerate(fin.readlines()):
                if skip_first and idx == 0: continue
                yield row
    
    p = Pool(24)
    iterator_evaluation_results = iterator(log_data_result_path, skip_first=True)
    partial_evaluation_fn = partial(validate_evaluation_results, threshold=config['threshold_wer'])
    validate_evaluation_results_map = p.imap_unordered(
        partial_evaluation_fn,
        tqdm(iterator_evaluation_results, desc="[Validate evaluation results]"),
        chunksize=10,
    )
    
    iterator_error_results = iterator(log_data_error_path, skip_first=False)
    partial_error_fn = partial(extract_error_results)
    extract_error_results_map = p.imap_unordered(
        partial_error_fn,
        tqdm(iterator_error_results, desc="[Validate error results]"),
        chunksize=10,
    )
    
    list_bad_audio = list()
    clean_data_manifests = []
    number_bad_audio = 0
    number_error_audio = 0
    number_evaluation_audio = 0
    
    for code, audio_path in validate_evaluation_results_map:
        if code != 0:
            list_bad_audio.append(audio_path)
            number_bad_audio += 1
        
    for audio_path in extract_error_results_map:
        list_bad_audio.append(audio_path)
        number_error_audio += 1
    
    # Extract audio from manifest except error and bad audio files
    with open(original_manifest_path, mode="r") as fin:
        for row in fin:
            row = row.replace("\n", "")
            json_row = json.loads(row)
            audio_path = json_row['audio_filepath']
            
            if audio_path not in list_bad_audio:
                clean_data_manifests.append(row)
        
            number_evaluation_audio += 1
        
    with open(manifest_path, mode='w', encoding='utf-8') as fout:
        data = '\n'.join(clean_data_manifests)
        fout.writelines(data + '\n')
        fout.close()
        
    LOGGER.log_info(f"Number of audio in evaluation manifest: {number_evaluation_audio}")
    LOGGER.log_info(f"Number of audio in clean evaluation manifest: {len(clean_data_manifests)}")
    LOGGER.log_info(f"Number of error audio: {number_error_audio}")
    LOGGER.log_info(f"Number of audio with wer >= {config['threshold_wer']}: {number_bad_audio}")

'''
python3 conformer_asr/evaluation/evaluate_asr_data.py -e \
    --original_manifest_path="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_original_train_manifest.json" \
    --dataset_name="vlsp2021_train" \
    --log_dir="/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data"
    
python3 conformer_asr/evaluation/evaluate_asr_data.py -c \
    --original_manifest_path="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_original_train_manifest.json" \
    --manifest_path="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_train_manifest.json" \
    --dataset_name="vlsp2021_train" \
    --log_dir="/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data" \
    --threshold_wer=0.2
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='', help='name of evaluation dataset')
    parser.add_argument('--log_dir', default='', help='name of evaluation dataset')
    parser.add_argument('--device', default='cuda', help='name of evaluation dataset')
    
    parser.add_argument('-e', '--evaluation', action='store_true', help='evaluate datasets')
    parser.add_argument('--original_manifest_path', default='', help='manifest path of evaluation dataset')

    parser.add_argument('-c', '--clean_evaluation_dataset', action='store_true', help='clean evaluate datasets')
    parser.add_argument('--manifest_path', default='', help='manifest path of evaluation dataset')
    parser.add_argument('--threshold_wer', default=0.2, help='name of evaluation dataset')
    args = parser.parse_args()
    
    evaluation_config = config.get_config(["evaluation"])
    
    if args.evaluation:
        if args.original_manifest_path == "" or args.dataset_name == "" or args.log_dir == "":
            LOGGER.log_error("Invalid evaluation configuration")
            exit(-1)
        
        evaluation_dataset_config = EvaluationConfig(
            original_manifest_path = args.original_manifest_path,
            dataset_name = args.dataset_name,
            log_dir = args.log_dir,
            device = args.device
        )   
        LOGGER.log_info("Inititalize Wav2Vec2 Model...")
        processor = Wav2Vec2Processor.from_pretrained(evaluation_config.model.wav2vec2_processor_path)
        model = Wav2Vec2ForCTC.from_pretrained(evaluation_config.model.wav2vec2_model_path).to(torch.device(args.device))
        ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, evaluation_config.model.lm_file_path)
        LOGGER.log_info("Done Inititalize Wav2Vec2 Model")
        print()
        
        evaluate_data(evaluation_dataset_config)
        
    elif args.clean_evaluation_dataset:
        if args.manifest_path == "" or args.original_manifest_path == ""\
            or args.log_dir == "" or args.dataset_name == "":
            LOGGER.log_error("Invalid clean dataset configuration")
            exit(-1)
        
        clean_dataset_config = CleanConfig(
            manifest_path = args.manifest_path,
            original_manifest_path = args.original_manifest_path,
            log_dir = args.log_dir,
            dataset_name = args.dataset_name,
            threshold_wer = args.threshold_wer,
        )
        clean_bad_data(clean_dataset_config)