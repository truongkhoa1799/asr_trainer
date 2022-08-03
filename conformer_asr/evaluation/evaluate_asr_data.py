import os
import csv
import copy
import json
import pathlib
import librosa
from typing import AnyStr, List
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

@dataclass
class EvaluationLog:
    """A configuration for the EvaluationLog.

    Attributes:
        audio_filepath: The title of the Menu.
        ground_truth_text: The body of the Menu.
        transcribed_text: The text for the button label.
        wer: Can it be cancelled?
    """
    audio_filepath: str
    ground_truth_text: str
    transcribed_text: str
    wer: int = 0

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
    # Evaluation dataset
    evaluatation_manifest_path = pathlib.Path(config.dataset)

    # Result
    dataset_name = config.result.dataset_name
    log_data_result_path = pathlib.Path(config.result.log_data_result_path)
    log_data_error_path = pathlib.Path(config.result.log_data_error_path)

    # Inititalize parameters
    log_data = []
    error_files = list()
    
    # Clear old log results
    if log_data_result_path.exists(): os.remove(str(log_data_result_path))
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
            
            log_data.append(evaluation_log.__dict__)
            
        except Exception as e:
            error_files.append(data["audio_filepath"])
        
        if idx % 500 == 0:
            Logger.log_info(f"Iteration {idx}")
            save_log_result(log_data, log_data_result_path)
            log_data = []
        
        # if idx >= 23: break

    if len(log_data) != 0:
        Logger.log_info("Save last data logs")
        save_log_result(log_data, log_data_result_path)
        log_data = []

def clean_bad_data(config):
    evaluatation_manifest_path = pathlib.Path(config.dataset)
    clean_evaluatation_manifest_path, ext = os.path.splitext(str(evaluatation_manifest_path))
    clean_evaluatation_manifest_path += f"_cleaned{ext}"
    if os.path.exists(clean_evaluatation_manifest_path):
        os.remove(clean_evaluatation_manifest_path)
    
    number_data_of_evaluatation_manifest = 0
    list_bad_audio = list()
    with open(config.result.log_data_result_path, mode="r") as f:
        for idx, row in enumerate(f.readlines()):
            if idx == 0: continue
            
            row = row.replace("\n", "")
            row_values = row.split(",")
            audio_path = row_values[0]
            wer = row_values[-1]

            if float(wer) >= float(config.threshold_wer):
                list_bad_audio.append(audio_path)
            
        f.close()
    
    clean_evaluatation_manifest = []
    with open(config.dataset, mode="r") as f:
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
    Logger.log_info(f"Number of audio with wer >= {config.threshold_wer}: {len(list_bad_audio)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', action='store_true', help='evaluate datasets')
    parser.add_argument('--clean_evaluation_dataset', action='store_true', help='clean evaluate datasets')
    args = parser.parse_args()
    
    evaluation_config = config.get_config(["evaluation"])
    
    if args.evaluation:
        Logger.log_info("Inititalize Wav2Vec2 Model...")
        
        processor = Wav2Vec2Processor.from_pretrained(evaluation_config.model.wav2vec2_processor_path)
        model = Wav2Vec2ForCTC.from_pretrained(evaluation_config.model.wav2vec2_model_path).to(torch.device('cuda'))
        ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, evaluation_config.model.lm_file_path)
        
        Logger.log_info("Done Inititalize Wav2Vec2 Model")
        evaluate_data(evaluation_config)
        
    elif args.clean_evaluation_dataset:
        clean_bad_data(evaluation_config)