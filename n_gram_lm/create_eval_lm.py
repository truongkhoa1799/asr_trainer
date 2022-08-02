import os
import json
import sys
PROJECT_DIR = "/".join(os.getcwd().split('/')[:-1])
sys.path.append(PROJECT_DIR)
from Conformer_ASR.scripts.utils import config, Logger, Config

LOGGER = Logger("CREATE__EVALUATE_BEAM_SEARCH_DECODING")
lm_config = config.get_config(["lm"])
train_manifest_cleaned = config.get_config(["training", "manifest", "train_manifest_cleaned"])
test_manifest_cleaned = config.get_config(["training", "manifest", "test_manifest_cleaned"])

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
    with open(manifest_path, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            data = json.loads(line)
            text_data.append(data["text"])

            if len(text_data) % 10000 == 0:
                LOGGER.log_info(f"\t\tSave {len(text_data)} text to files")
                save_text_file(data=text_data, text_path=text_file)
                text_data = []
        f.close()
        
    if len(text_data) != 0:
        save_text_file(data=text_data, text_path=text_file)
        LOGGER.log_info(f"\t\tSave {len(text_data)} text to files")
        text_data = []
    
def create_lm_model():
    LOGGER.log_info("Start creating Beam Search Decoding")
    
    LOGGER.log_info("\tExtract text in training and testing manifest")
    os.system(f"cat {train_manifest_cleaned} {test_manifest_cleaned} > {lm_config.data.train_test_manifest}")
    create_text_file_from_manifest(lm_config.data.train_test_manifest, lm_config.data.manifest_data)
    
    LOGGER.log_info("\tConcate Manifest data and Assistant data")
    os.system(f"cat {lm_config.data.manifest_data} {lm_config.data.assistant_data} > {lm_config.data.all_data}")
    num_manifest_data = os.popen(f"wc -l {lm_config.data.manifest_data}").read()
    num_assistant_data = os.popen(f"wc -l {lm_config.data.assistant_data}").read()
    num_all_data = os.popen(f"wc -l {lm_config.data.all_data}").read()
    LOGGER.log_info(f"\t\t{num_manifest_data.strip()}")
    LOGGER.log_info(f"\t\t{num_assistant_data.strip()}")
    LOGGER.log_info(f"\t\t{num_all_data.strip()}")
    
    run_script = f"python3 {lm_config.kenlm.train_kenlm} \
    --nemo_model_file {lm_config.model.asr_path} \
    --train_file {lm_config.data.all_data} \
    --kenlm_bin_path {lm_config.kenlm.kenlm_bin} \
    --kenlm_model_file {lm_config.model.kenlm} \
    --ngram_length {lm_config.model.ngram_length}"

    LOGGER.log_info("\tStart training Beamserach Model")
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
    create_lm_model()
    # eval_lm_model()