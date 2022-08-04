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
Logger = Logger(name="EVALUATING_ASR_DATA")

class EvaluationLog(TypedDict):
    audio_filepath: str
    ground_truth_text: str
    transcribed_text: str
    wer: int = 0

class EvaluationConfig(TypedDict):
    config
    manifest_path: str
    dataset_name: str

class CleanConfig(TypedDict):
    config
    log_data_result_path: str
    threshold_wer: float
    clean_manifest_path: str
    manifest_path: str

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


def transcribe_ASR(raw_signal):
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
    ).input_values.to("cuda")
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
    dataset_name = config['dataset_name']
    evaluatation_manifest_path = Path(config['manifest_path'])
    result_directory = Path(config['config'].result_directory)    
    evaluation_dataset_result_directory = result_directory.joinpath(dataset_name)
    if not evaluation_dataset_result_directory.exists():
        evaluation_dataset_result_directory.mkdir(parents=True, exist_ok=True)
    
    log_data_result_path = evaluation_dataset_result_directory.joinpath(f"{dataset_name}_result.log")
    log_data_error_path = evaluation_dataset_result_directory.joinpath(f"{dataset_name}_error.log")

    # Inititalize parameters
    log_data = []
    error_files = list()
    
    # Clear old log results
    if log_data_result_path.exists(): 
        log_data_error_path.unlink(missing_ok=True)
    if log_data_result_path.exists(): 
        log_data_error_path.unlink(missing_ok=True)
        
    manifest_data, _ = read_manifest(str(evaluatation_manifest_path))
    for idx, data in enumerate(manifest_data):
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
        
        if idx % 500 == 0:
            Logger.log_info(f"Iteration {idx}")
            save_log_result(log_data, log_data_result_path)
            log_data = []
        
        # if idx >= 14: break

    if len(log_data) != 0:
        Logger.log_info("Save last data logs")
        save_log_result(log_data, log_data_result_path)
        log_data = []
    
    # Save error file
    with open(log_data_error_path, mode='w', encoding='utf-8') as fout:
        error_result = "\n".join(error_files)
        fout.writelines(error_result + '\n')
        fout.close()
    
def clean_bad_data(config):
    log_data_result_path = Path(config['log_data_result_path'])
    if not log_data_result_path.exists():
        Logger.log_error("Invalid log_data_result_path")
        return
    
    dataset_name = str(log_data_result_path.parent.name)
    clean_evaluatation_manifest_path = Path(config['clean_manifest_path'])
    if clean_evaluatation_manifest_path.exists():
        clean_evaluatation_manifest_path.unlink(missing_ok=True)
    
    number_data_of_evaluatation_manifest = 0
    list_bad_audio = list()
    with open(log_data_result_path, mode="r") as f:
        for idx, row in enumerate(f.readlines()):
            if idx == 0: continue
            
            row = row.replace("\n", "")
            row_values = row.split(",")
            audio_path = row_values[0]
            wer = row_values[-1]

            if float(wer) >= float(config['threshold_wer']):
                list_bad_audio.append(audio_path)
            
        f.close()
    
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
        
    Logger.log_info(f"Number of audio in evaluation manifest: {number_data_of_evaluatation_manifest}")
    Logger.log_info(f"Number of audio in clean evaluation manifest: {len(clean_evaluatation_manifest)}")
    Logger.log_info(f"Number of audio with wer >= {config['threshold_wer']}: {len(list_bad_audio)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation', action='store_true', help='evaluate datasets')
    parser.add_argument('--manifest_path', default='', help='manifest path of evaluation dataset')
    parser.add_argument('--dataset_name', default='', help='name of evaluation dataset')

    parser.add_argument('-c', '--clean_evaluation_dataset', action='store_true', help='clean evaluate datasets')
    parser.add_argument('--log_data_result_path', default='', help='log result of evaluation step')
    parser.add_argument('--clean_manifest_path', default='', help='clean manifest path of evaluation dataset')
    parser.add_argument('--threshold_wer', default=0.2, help='name of evaluation dataset')
    args = parser.parse_args()
    
    evaluation_config = config.get_config(["evaluation"])
    
    if args.evaluation:
        if args.manifest_path == "" or args.dataset_name == "":
            Logger.log_error("Invalid evaluation configuration")
            exit(-1)
        
        evaluation_dataset_config = EvaluationConfig(
            config = evaluation_config,
            manifest_path = args.manifest_path,
            dataset_name = args.dataset_name
        )   
        Logger.log_info("Inititalize Wav2Vec2 Model...")
        processor = Wav2Vec2Processor.from_pretrained(evaluation_config.model.wav2vec2_processor_path)
        model = Wav2Vec2ForCTC.from_pretrained(evaluation_config.model.wav2vec2_model_path).to(torch.device('cuda'))
        ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, evaluation_config.model.lm_file_path)
        Logger.log_info("Done Inititalize Wav2Vec2 Model")
        
        evaluate_data(evaluation_dataset_config)
        
    elif args.clean_evaluation_dataset:
        if args.log_data_result_path == "" or args.clean_manifest_path == ""\
            or args.manifest_path == "":
            Logger.log_error("Invalid clean dataset configuration")
            exit(-1)
        
        clean_dataset_config = CleanConfig(
            config = evaluation_config,
            log_data_result_path = args.log_data_result_path,
            threshold_wer = args.threshold_wer,
            clean_manifest_path = args.clean_manifest_path,
            manifest_path = args.manifest_path
        )
        clean_bad_data(clean_dataset_config)