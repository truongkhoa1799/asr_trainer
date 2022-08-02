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

import torch
import kenlm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

from jiwer import wer
from scripts.utils import Logger, read_manifest
Logger = Logger(name="EVALUATING_ASR_DATA")

# evaluatated_manifest_path = pathlib.Path("/home/khoatlv/data/infore/infore_415h/manifests/infore_415h_manifest.json")
evaluatated_manifest_path = pathlib.Path("/home/khoatlv/data/data_collected/Zalo/manifests/manifests.json")

# Result
dataset_name = "Infore_415h"
log_data_result_path = pathlib.Path(f"/home/khoatlv/Conformer_ASR/scripts/evaluation/results/ASR_data/{dataset_name}_result.log")
log_data_error_path = pathlib.Path(f"/home/khoatlv/Conformer_ASR/scripts/evaluation/results/ASR_data/{dataset_name}_error.log")

# Model Config
wav2vec2_processor_path = "/home/khoatlv/Conformer_ASR/scripts/evaluation/wav2vec_models/preprocessor"
wav2vec2_model_path = "/home/khoatlv/Conformer_ASR/scripts/evaluation/wav2vec_models/CTCModel"
lm_file_path = "/home/khoatlv/Conformer_ASR/scripts/evaluation/wav2vec_models/4-gram-lm_large.bin"

# Init ASR model
processor = None
model = None
ngram_lm_model = None

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

def save_log_result(log_data):
    fieldnames = ['audio_filepath', 'ground_truth_text', 'transcribed_text', 'wer']
    datas = list(map(lambda log: [*log], log_data))
    print(log_data)
    print(datas)

    f = open(str(log_data_result_path), 'a', encoding='UTF8', newline='')
    writer = csv.writer(f)
    if not os.path.exists(str(log_data_result_path)):
        writer.writerow(fieldnames)
    writer.writerows(datas)
    f.close()
            

def evaluate_data(manifest_path: pathlib.Path):
    # Inititalize parameters
    log_data = []
    error_files = list()
    
    # Clear old log results
    if log_data_result_path.exists(): os.remove(str(log_data_result_path))
    manifest_data, _ = read_manifest(str(manifest_path))
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
        
        if idx % 1000 == 0:
            Logger.log_info(f"Iteration {idx}")
            save_log_result(log_data)
            log_data = []
        break

    if len(log_data) != 0:
        Logger.log_info("Save last data logs")
        save_log_result(log_data)
        log_data = []

if __name__ == "__main__":
    Logger.log_info("Inititalize Wav2Vec2 Model...")
    processor = Wav2Vec2Processor.from_pretrained(wav2vec2_processor_path)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_path).to(torch.device('cuda'))
    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_file_path)
    Logger.log_info("Done Inititalize Wav2Vec2 Model")
    
    evaluate_data(evaluatated_manifest_path)