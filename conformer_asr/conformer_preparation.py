import os
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as ptl
from nemo.utils import exp_manager
import torch.nn as nn
import torch
from tqdm.auto import tqdm

WANDB_LOGGER = True
ASR_DIR = "/home/khoatlv/Conformer_ASR"

VOCAB_SIZE = 256
tokenizer_dir = os.path.join(ASR_DIR, "tokenizers", "tokenizers_conformer")
TOKENIZER_TYPE = "spe"  # can be wpe or spe
SPE_TYPE = "bpe"        # ["bpe", "unigram"]

model_config = "config/conformer_small_ctc_bpe.yaml"
config_path = os.path.join(ASR_DIR, model_config)

train_manifest_cleaned = "/home/khoatlv/manifests/train_manifest_processed.json"
test_manifest_cleaned = "/home/khoatlv/manifests/test_manifest_processed.json"

# Tokenizer path
if TOKENIZER_TYPE == 'spe':
  TOKENIZER_TYPE_CFG = "bpe"
  TOKENIZER_DIR = os.path.join(tokenizer_dir, f"tokenizer_spe_{TOKENIZER_TYPE_CFG}_v{VOCAB_SIZE}")
else:
  TOKENIZER_TYPE_CFG = "wpe"
  TOKENIZER_DIR = os.path.join(tokenizer_dir, f"tokenizer_wpe_v{VOCAB_SIZE}")

print("Tokenizer directory :", TOKENIZER_DIR)

def load_config(path):
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    config.model.train_ds.manifest_filepath = train_manifest_cleaned
    config.model.validation_ds.manifest_filepath = test_manifest_cleaned
    config.model.test_ds.manifest_filepath = test_manifest_cleaned
    
    return config

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path="/home/khoatlv/Conformer_ASR/models/conformer/Conformer_small_epoch=98.nemo",
    map_location='cuda'
)

config = load_config(model_config)

# Set tokenizer config
asr_model.cfg.tokenizer.dir = TOKENIZER_DIR
asr_model.cfg.tokenizer.type = TOKENIZER_TYPE_CFG

asr_model.setup_training_data(config.model.train_ds)
asr_model.setup_validation_data(config.model.validation_ds)
asr_model.setup_multiple_test_data(config.model.test_ds)

with open_dict(asr_model.cfg):
    asr_model.cfg.optim = config.model.optim
    asr_model.cfg.spec_augment = config.model.spec_augment    
    
asr_model.spec_augmentation = asr_model.from_config_dict(config.model.spec_augment)
asr_model.setup_optimization(config.model.optim)

asr_model._wer.use_cer = True
asr_model._wer.log_prediction = True

trainer = ptl.Trainer(**config.trainer)
asr_model.set_trainer(trainer)
asr_model.cfg = asr_model._cfg

exp_config = exp_manager.ExpManagerConfig(**config.exp_manager)
exp_config = OmegaConf.structured(exp_config)
logdir = exp_manager.exp_manager(trainer, exp_config)

def analyse_ctc_failures_in_model(model):
    count_ctc_failures = 0
    am_seq_lengths = []
    target_seq_lengths = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    mode = model.training
    
    train_dl = model.train_dataloader()

    with torch.no_grad():
      model = model.eval()
      for batch in tqdm(train_dl, desc='Checking for CTC failures'):
          x, x_len, y, y_len = batch
          x, x_len = x.to(device), x_len.to(device)
          x_logprobs, x_len, greedy_predictions = model(input_signal=x, input_signal_length=x_len)

          # Find how many CTC loss computation failures will occur
          for xl, yl in zip(x_len, y_len):
              if xl <= yl:
                  count_ctc_failures += 1

          # Record acoustic model lengths=
          am_seq_lengths.extend(x_len.to('cpu').numpy().tolist())

          # Record target sequence lengths
          target_seq_lengths.extend(y_len.to('cpu').numpy().tolist())
          
          del x, x_len, y, y_len, x_logprobs, greedy_predictions
    
    if mode:
      model = model.train()
      
    return count_ctc_failures, am_seq_lengths, target_seq_lengths

results = analyse_ctc_failures_in_model(asr_model)
num_ctc_failures, am_seq_lengths, target_seq_lengths = results
if num_ctc_failures > 0:
  print(f"\nCTC loss will fail for {num_ctc_failures} samples ({num_ctc_failures * 100./ float(len(am_seq_lengths))} % of samples)!\n"
                  f"Increase the vocabulary size of the tokenizer so that this number becomes close to zero !")
else:
  print("No CTC failure cases !")
# Compute average ratio of T / U
avg_T = sum(am_seq_lengths) / float(len(am_seq_lengths))
avg_U = sum(target_seq_lengths) / float(len(target_seq_lengths))

avg_length_ratio = 0
for am_len, tgt_len in zip(am_seq_lengths, target_seq_lengths):
  avg_length_ratio += (am_len / float(tgt_len))
avg_length_ratio = avg_length_ratio / len(am_seq_lengths)

print(f"Average Acoustic model sequence length = {avg_T}")
print(f"Average Target sequence length = {avg_U}")
print()
print(f"Ratio of Average AM sequence length to target sequence length = {avg_length_ratio}")
