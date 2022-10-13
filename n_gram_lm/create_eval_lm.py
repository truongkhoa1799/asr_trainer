import os
import json
import sys
import argparse
from pathlib import Path
from typing import TypedDict
from conformer_asr.utils import config, Logger, Config

'''
python3 /home/khoatlv/ASR_Nemo/n_gram_lm/create_eval_lm.py -c
'''

LOGGER = Logger("CREATE__EVALUATE_BEAM_SEARCH_DECODING")
class CreateLMConfig(TypedDict):
    lm_config: str
    train_manifest_cleaned: str
    test_manifest_cleaned: str

def save_text_file(data, text_path):
    data = "\n".join(data)
    mode=None
    if os.path.exists(text_path): mode = 'a'
    else: mode = 'w'

    with open(text_path, mode, encoding='UTF8', newline='') as f:
        f.writelines(data + '\n')
        f.close()


def create_text_file_from_manifest(manifest_path, text_file):
    if os.path.exists(text_file): os.remove(text_file)
    text_data = []
    count = 0
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            data = json.loads(line)
            text_data.append(data["text"])

            if len(text_data) % 10000 == 0:
                count += 10000
                LOGGER.log_info(f"\t\tSave {count} text to files")
                save_text_file(data=text_data, text_path=text_file)
                text_data = []
        
    if len(text_data) != 0:
        save_text_file(data=text_data, text_path=text_file)
        LOGGER.log_info(f"\t\tSave {len(text_data)} text to files")
        text_data = []
    
def create_lm_model(config):
    LOGGER.log_info("Start creating Beam Search Decoding")
    
    if config['lm_config'].data.use_collected_data:
        LOGGER.log_info("\tExtract texts in collected data directory to collected_data.txt")
        collected_data_dir = Path(config['lm_config'].data.collected_data_dir)
        collected_datas = list(collected_data_dir.glob("*.txt"))
        script = f"cat "
        for collected_data in collected_datas:
            script += f"{collected_data.resolve()} "
            
        script += f"> {config['lm_config'].data.collected_data}"
        os.system(script)
    
    LOGGER.log_info("\tExtract text in training and testing manifest to train_test_manifest")
    os.system(f"cat {config['train_manifest_cleaned']} {config['test_manifest_cleaned']} > {config['lm_config'].data.train_test_manifest}")
    create_text_file_from_manifest(config['lm_config'].data.train_test_manifest, config['lm_config'].data.manifest_data)
    
    LOGGER.log_info("\tConcate Manifest data and Assistant data and collected data")
    script = f"cat {config['lm_config'].data.manifest_data} "
    if config['lm_config'].data.use_collected_data:
        script += f"{config['lm_config'].data.collected_data} "
        
    if config['lm_config'].data.use_assistant_data:
        script += f"{config['lm_config'].data.assistant_data} "
        
    script += f"> {config['lm_config'].data.all_data}"
    os.system(script)
    
    num_manifest_data = os.popen(f"wc -l {config['lm_config'].data.manifest_data}").read()
    num_assistant_data = os.popen(f"wc -l {config['lm_config'].data.assistant_data}").read()
    num_all_data = os.popen(f"wc -l {config['lm_config'].data.all_data}").read()
    
    LOGGER.log_info(f"\t\t{num_manifest_data.strip()}")
    LOGGER.log_info(f"\t\t{num_all_data.strip()}")
    
    if config['lm_config'].data.use_collected_data:
        num_collected_data = os.popen(f"wc -l {config['lm_config'].data.collected_data}").read()
        LOGGER.log_info(f"\t\t{num_collected_data.strip()}")
        
    if config['lm_config'].data.use_assistant_data:
        num_assistant_data = os.popen(f"wc -l {config['lm_config'].data.assistant_data}").read()
        LOGGER.log_info(f"\t\t{num_assistant_data.strip()}")
    
    run_script = f"python3 {config['lm_config'].kenlm.train_kenlm} \
    --nemo_model_file {config['lm_config'].model.asr_path} \
    --train_file {config['lm_config'].data.all_data} \
    --kenlm_bin_path {config['lm_config'].kenlm.kenlm_bin} \
    --kenlm_model_file {config['lm_config'].model.kenml_model_file} \
    --ngram_length {config['lm_config'].model.ngram_length}"

    LOGGER.log_info("\tStart training Beamserach Model")
    # print(run_script)
    os.system(run_script)
    print()

def eval_lm_model():
    beam_width = "64 128"
    beam_alpha = "1.5 2.0 2.5"
    beam_beta = "1.5 2.0"

    EVAL_BEAMSEARCH_DIRS = "/home/khoatlv/Conformer_ASR/n_gram_lm/eval_beamsearch"
    LOG_PROBS_CACHE = "/home/khoatlv/Conformer_ASR/n_gram_lm/eval_beamsearch/probs_cache_file"
    LOGGER.log_info(f"Start Evaludating Beam Search Decoding beam_width with:\
        {beam_width} beam_alpha: {beam_alpha} beam_beta: {beam_beta} and store in {EVAL_BEAMSEARCH_DIRS}")
    
    eval_script = f"python3 {lm_config.kenlm.eval_ngram} \
                    --nemo_model_file {lm_config.model.asr_path} \
                    --input_manifest {lm_config.data.train_test_manifest} \
                    --kenlm_model_file {lm_config.model.kenml_model_file} \
                    --acoustic_batch_size 16 \
                    --beam_width {beam_width} \
                    --beam_alpha {beam_alpha} \
                    --beam_beta {beam_beta} \
                    --preds_output_folder {EVAL_BEAMSEARCH_DIRS} \
                    --decoding_mode beamsearch_ngram \
                    --device cuda \
                    --probs_cache_file {LOG_PROBS_CACHE} > /home/khoatlv/Conformer_ASR/n_gram_lm/eval_beamsearch/log.txt"
                    
    os.system(eval_script)
    LOGGER.log_info(f"Finish Evaludating Beam Search Decoding")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--create', action='store_true', help='create language model')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate language model')
    args = parser.parse_args()
    
    lm_config = config.get_config(["lm"])
    train_manifest_cleaned = config.get_config(["training", "manifest", "train_manifest_cleaned"])
    test_manifest_cleaned = config.get_config(["training", "manifest", "test_manifest_cleaned"])
    
    if args.create:
        create_lm_config = CreateLMConfig(
            lm_config=lm_config,
            train_manifest_cleaned=train_manifest_cleaned,
            test_manifest_cleaned=test_manifest_cleaned
        )
        create_lm_model(create_lm_config)
    elif args.evaluate:
        eval_lm_model()