import os
from datetime import datetime
from pathlib import Path
from conformer_asr.utils import config, Logger

LOGGER = Logger(name="CREATE_TOKENIZER")

'''
python3 conformer_asr/tokenizer/create_tokenizer.py
'''

if __name__ == "__main__":
    config = config.get_config(["training"])
    tokenizer_dir = Path(config.tokenizer.tokenizer_dir)
    if not tokenizer_dir.exists():
        tokenizer_dir.mkdir(parents=True, recursive=True)
    
    current_time = str(datetime.now()).split()[0]
    tokenizer_conformer_dir = tokenizer_dir.joinpath(config.tokenizer.tokenizer_conformer + '_' + current_time)
    vocab_size = config.vocab_size
    tokenizer_type = config.tokenizer.type  # can be wpe or spe
    tokenizer_type_cfg = config.tokenizer.type_cfg        # ["bpe", "unigram"]
    
    cleaned_train_manifest = config.manifest.train_manifest_cleaned
    cleaned_test_manifest = config.manifest.test_manifest_cleaned
    
    num_train_data = os.popen(f"wc -l {cleaned_train_manifest}").read().strip()
    num_test_data = os.popen(f"wc -l {cleaned_test_manifest}").read().strip()
    
    LOGGER.log_info(f"Training data\t\t: {num_train_data}")
    LOGGER.log_info(f"Testing data\t\t: {num_test_data}")
    LOGGER.log_info(f"Tokenizer type\t\t: {tokenizer_type}")
    LOGGER.log_info(f"Tokenizer type config\t: {tokenizer_type_cfg}")
    LOGGER.log_info(f"Tokenizer directory\t: {tokenizer_conformer_dir}")
    
    python_script = f"python3 conformer_asr/tokenizer/process_asr_text_tokenizer.py \
        --manifest={cleaned_train_manifest} \
        --data_root={tokenizer_conformer_dir} \
        --tokenizer={tokenizer_type} \
        --spe_type={tokenizer_type_cfg} \
        --spe_character_coverage=1.0 \
        --no_lower_case \
        --log \
        --vocab_size={vocab_size}"
        
    result = os.popen(python_script).read()
    print(result)