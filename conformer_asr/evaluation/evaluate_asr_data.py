import os
import csv
import copy
import json
from pathlib import Path
import librosa
from typing import AnyStr, List, TypedDict
import joblib
from tqdm.auto import tqdm
from joblib.parallel import Parallel
from dataclasses import dataclass

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
    manifest_path: str
    log_dir: str
    dataset_name: str

class CleanConfig(TypedDict):
    config
    manifest_path: str
    clean_manifest_path: str
    log_data_result_path: str
    log_data_error_path: str
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
    evaluatation_manifest_path = Path(config['manifest_path'])
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
            transcribed_text = transcribe_ASR(signal)
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
    
def clean_bad_data(config):
    manifest_path = Path(config['manifest_path'])
    clean_evaluatation_manifest_path = Path(config['clean_manifest_path'])
    
    log_data_result_path = Path(config['log_data_result_path'])
    log_data_error_path = Path(config['log_data_error_path'])
    
    dataset_name = str(log_data_result_path.parent.name)
    
    if not log_data_result_path.exists() or not log_data_error_path.exists():
        LOGGER.log_error("Invalid log_data_result_path or log_data_error_path")
        return
    
    if not manifest_path.exists():
        LOGGER.log_error("Invalid manifest_path")
        return
    
    if clean_evaluatation_manifest_path.exists():
        clean_evaluatation_manifest_path.unlink(missing_ok=True)
        
    list_bad_audio = list()
    number_bad_audio = 0
    number_error_audio = 0
    number_data_of_evaluatation_manifest = 0
    
    # Read list bad audio files
    with open(log_data_result_path, mode="r") as f:
        for idx, row in enumerate(f.readlines()):
            if idx == 0: continue
            
            row = row.replace("\n", "")
            row_values = row.split(",")
            audio_path = row_values[0]
            wer = row_values[-1]

            if float(wer) >= float(config['threshold_wer']):
                list_bad_audio.append(audio_path)
                number_bad_audio += 1
            
        f.close()
    
    # Read list error files
    with open(log_data_error_path, mode="r") as f:
        for audio_path in f.readlines():
            audio_path = audio_path.replace("\n", "")
            list_bad_audio.append(audio_path)
            number_error_audio += 1
    
    # Extract audio from manifest except error and bad audio files
    clean_evaluatation_manifest = []
    with open(config['manifest_path'], mode="r") as f:
        for row in f.readlines(): 
            row = row.replace("\n", "")
            json_row = json.loads(row)
            audio_path = json_row['audio_filepath']
            
            if audio_path not in list_bad_audio:
                clean_evaluatation_manifest.append(row)
        
            number_data_of_evaluatation_manifest += 1
        
        f.close()
        
    with open(clean_evaluatation_manifest_path, mode='w', encoding='utf-8') as fout:
        data = '\n'.join(clean_evaluatation_manifest)
        fout.writelines(data + '\n')
        fout.close()
        
    LOGGER.log_info(f"Number of audio in evaluation manifest: {number_data_of_evaluatation_manifest}")
    LOGGER.log_info(f"Number of audio in clean evaluation manifest: {len(clean_evaluatation_manifest)}")
    LOGGER.log_info(f"Number of error audio: {number_error_audio}")
    LOGGER.log_info(f"Number of audio with wer >= {config['threshold_wer']}: {number_bad_audio}")

'''
python3 conformer_asr/evaluation/evaluate_asr_data.py -e \
    --manifest_path="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_original_train_manifest.json" \
    --dataset_name="vlsp2021_train" \
    --log_dir="/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data"
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation', action='store_true', help='evaluate datasets')
    parser.add_argument('--manifest_path', default='', help='manifest path of evaluation dataset')
    parser.add_argument('--dataset_name', default='', help='name of evaluation dataset')
    parser.add_argument('--log_dir', default='', help='name of evaluation dataset')
    parser.add_argument('--device', default='cpu', help='name of evaluation dataset')

    parser.add_argument('-c', '--clean_evaluation_dataset', action='store_true', help='clean evaluate datasets')
    parser.add_argument('--log_data_result_path', default='', help='log result of evaluation step')
    parser.add_argument('--log_data_error_path', default='', help='log error audio of evaluation step')
    parser.add_argument('--clean_manifest_path', default='', help='clean manifest path of evaluation dataset')
    parser.add_argument('--threshold_wer', default=0.2, help='name of evaluation dataset')
    args = parser.parse_args()
    
    evaluation_config = config.get_config(["evaluation"])
    
    if args.evaluation:
        if args.manifest_path == "" or args.dataset_name == "" or args.log_dir == "":
            LOGGER.log_error("Invalid evaluation configuration")
            exit(-1)
        
        evaluation_dataset_config = EvaluationConfig(
            manifest_path = args.manifest_path,
            dataset_name = args.dataset_name,
            log_dir = args.log_dir
        )   
        LOGGER.log_info("Inititalize Wav2Vec2 Model...")
        processor = Wav2Vec2Processor.from_pretrained(evaluation_config.model.wav2vec2_processor_path)
        model = Wav2Vec2ForCTC.from_pretrained(evaluation_config.model.wav2vec2_model_path).to(torch.device(args.device))
        ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, evaluation_config.model.lm_file_path)
        LOGGER.log_info("Done Inititalize Wav2Vec2 Model")
        print()
        
        evaluate_data(evaluation_dataset_config)
        
    elif args.clean_evaluation_dataset:
        if args.log_data_result_path == "" or args.clean_manifest_path == ""\
            or args.manifest_path == "" or args.log_data_error_path == "":
            LOGGER.log_error("Invalid clean dataset configuration")
            exit(-1)
        
        clean_dataset_config = CleanConfig(
            config = evaluation_config,
            manifest_path = args.manifest_path,
            clean_manifest_path = args.clean_manifest_path,
            log_data_result_path = args.log_data_result_path,
            log_data_error_path = args.log_data_error_path,
            threshold_wer = args.threshold_wer,
        )
        clean_bad_data(clean_dataset_config)