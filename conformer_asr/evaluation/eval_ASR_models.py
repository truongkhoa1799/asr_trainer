import nemo.collections.asr as nemo_asr
import os
from tqdm.auto import tqdm
import numpy as np
import json
import pickle
from nemo.utils import logging
import torch
import contextlib
import nemo
import editdistance
import csv

test_manifest = "/home/khoatlv/manifests/test_manifest_processed.json"
train_manifest = "/home/khoatlv/manifests/train_manifest_processed.json"
all_data_manifest = "/home/khoatlv/Conformer_ASR/scripts/evaluation/all_data_manifest.json"

# pickle data
probs_cache_file = "/home/khoatlv/Conformer_ASR/scripts/evaluation/eval_asr_model/probs_cache_file"
conformer_transcribe_log = "/home/khoatlv/Conformer_ASR/scripts/evaluation/eval_asr_model/conformer_log.json"

# Conformer config
lm_path = "/home/khoatlv/Conformer_ASR/n_gram_lm/n_gram_lm_model/6-conformer-small-gram_trained.bin"
asr_model_path = "/home/khoatlv/Conformer_ASR/models/conformer/Conformer_small_epoch=98.nemo"

os.system(f"cat {test_manifest} {train_manifest} > {all_data_manifest}")
os.system(f"wc -l {all_data_manifest}")

use_amp = True
acoustic_batch_size = 16
beam_width = 200
alpha=2
beta=2.5

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

# Restore ASR Model
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=asr_model_path,
    map_location='cuda'    
)

# Restore Beam Search N-LM
TOKEN_OFFSET = 100
vocab = asr_model.decoder.vocabulary
vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
ids_to_text_func = asr_model.tokenizer.ids_to_text

beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
    vocab=list(vocab),
    beam_width=beam_width,
    alpha=alpha, 
    beta=beta,
    lm_path=lm_path,
    num_cpus=max(os.cpu_count(), 1),
    input_tensor=False
)

def eval_comformer():
    # Load manifest data and extract audio_path, target text
    target_transcripts = []
    with open(all_data_manifest, 'r') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {all_data_manifest} ...", ncols=120):
            data = json.loads(line)
            target_transcripts.append(data['text'])
            audio_file_paths.append(data['audio_filepath'])
    
    # Load audio wav and transribe
    if probs_cache_file and os.path.exists(probs_cache_file):
        logging.info(f"Found a pickle file of probabilities at '{probs_cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{probs_cache_file}' ...")
        with open(probs_cache_file, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{probs_cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:
        if use_amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast
        else:

            @contextlib.contextmanager
            def autocast():
                yield

        with autocast():
            with torch.no_grad():
                all_logits = asr_model.transcribe(audio_file_paths, batch_size=acoustic_batch_size, logprobs=True)
        all_probs = [softmax(logits) for logits in all_logits]
        if probs_cache_file:
            logging.info(f"Writing pickle files of probabilities at '{probs_cache_file}'...")
            with open(probs_cache_file, 'wb') as f_dump:
                pickle.dump(all_probs, f_dump)
                
    logging.info(f"==============================Starting the beam search decoding===============================")
    # logging.info(f"Grid search size: {len([]])}")
    logging.info(f"It may take some time...")
    logging.info(f"==============================================================================================")
    
    wer_dist_count = 0
    words_count = 0
    sample_idx = 0
    
    if conformer_transcribe_log:
        out_file = open(conformer_transcribe_log, 'w', encoding='UTF8', newline='')
        writer = csv.writer(out_file)
        headers = ["audio_filepath", "pred_text", "reference", "wer"]
        writer.writerow(headers)
    
    it = tqdm(
        range(int(np.ceil(len(all_probs) / acoustic_batch_size))),
        desc=f"Beam search decoding with width={beam_width}, alpha={alpha}, beta={beta}",
        ncols=120,
    )
    for batch_idx in it:
        # disabling type checking
        with nemo.core.typecheck.disable_checks():
            probs_batch = all_probs[batch_idx * acoustic_batch_size : (batch_idx + 1) * acoustic_batch_size]
            beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)
        
        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            words_count += len(target_split_w)
            
            # For BPE encodings, need to shift by TOKEN_OFFSET to retrieve the original sub-word ids
            pred_text = ids_to_text_func([ord(c) - TOKEN_OFFSET for c in beams[0][1]])
            pred_split_w = pred_text.split()
            wer_dist = editdistance.eval(target_split_w, pred_split_w)
            wer_dist_count += wer_dist
            
            wer = round(float("{:.2}".format(wer_dist / len(target_split_w))), 2)
            audio_path = audio_file_paths[sample_idx + beams_idx]
            if round(float(wer), 2) > 0.2: writer.writerow([audio_path, pred_text, target, wer])
            # print(f"target: {target}. pred_text: {pred_text}. wer: {wer}%")
        # break
        
        sample_idx += len(probs_batch)
    
    logging.info(
        'WER with beam search decoding and N-gram model = {:.2}'.format(wer_dist_count / words_count))
    
    if conformer_transcribe_log:
        out_file.close()

if __name__ == "__main__": 
    eval_comformer()