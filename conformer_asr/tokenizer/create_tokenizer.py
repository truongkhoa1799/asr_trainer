import os
import time
from conformer_asr.utils import config, Logger

LOGGER = Logger(name="CREATE_TOKENIZER")

if __name__ == "__main__":
    config = config.get_config(["training"])
    tokenizer_dir = config.tokenizer.tokenizer_dir.split("_")
    tokenizer_dir = "_".join([*tokenizer_dir[:-1], str(int(round(time.time(), 0)))])
    vocab_size = config.vocab_size

    tokenizer_cfg_type = config.tokenizer.type  # can be wpe or spe
    type_cfg = config.tokenizer.type_cfg        # ["bpe", "unigram"]
    
    cleaned_train_manifest = config.manifest.train_manifest_cleaned
    cleaned_test_manifest = config.manifest.test_manifest_cleaned
    
    num_train_data = os.popen(f"wc -l {cleaned_train_manifest}").read().strip()
    num_test_data = os.popen(f"wc -l {cleaned_test_manifest}").read().strip()
    
    LOGGER.log_info(f"Training data: {num_train_data}")
    LOGGER.log_info(f"Testing data: {num_test_data}")
    LOGGER.log_info(f"Tokenizer type: {tokenizer_cfg_type}")
    LOGGER.log_info(f"Tokenizer type config: {type_cfg}")
    LOGGER.log_info(f"Tokenizer directory: {tokenizer_dir}")
    LOGGER.log_info(f"Tokenizer Conformer: {config.tokenizer.tokenizer_conformer}")
    
    python_script = f"python3 conformer_asr/tokenizer/process_asr_text_tokenizer.py \
        --manifest={cleaned_train_manifest} \
        --data_root={tokenizer_dir} \
        --tokenizer={tokenizer_cfg_type} \
        --spe_type={type_cfg} \
        --spe_character_coverage=1.0 \
        --no_lower_case \
        --log \
        --vocab_size={vocab_size}"
        
    result = os.popen(python_script).read()
    print(result)