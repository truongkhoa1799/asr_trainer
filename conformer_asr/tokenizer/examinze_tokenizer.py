import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import pytorch_lightning as ptl
from nemo.utils import exp_manager
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict

from conformer_asr.utils import config, Logger

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

if __name__ == '__main__':
    print("Start examize tokenizer config")
    config = config.get_config(["training"])
    
    vocab_size = config.vocab_size
    tokenizer_dir = config.tokenizer.tokenizer_dir
    tokenizer_type = config.tokenizer.type  # can be wpe or spe
    tokenizer_type_cfg = config.tokenizer.type_cfg        # ["bpe", "unigram"]
    
    cleaned_train_manifest = config.manifest.train_manifest_cleaned
    cleaned_test_manifest = config.manifest.test_manifest_cleaned
    
    num_train_data = os.popen(f"wc -l {cleaned_train_manifest}").read().strip()
    num_test_data = os.popen(f"wc -l {cleaned_test_manifest}").read().strip()
    
    print(f"Training data: {num_train_data}")
    print(f"Testing data: {num_test_data}")
    print(f"Tokenizer type: {tokenizer_type}")
    print(f"Tokenizer type config: {tokenizer_type_cfg}")
    print(f"Tokenizer directory: {tokenizer_dir}")
    print(f"Tokenizer Conformer: {config.tokenizer.tokenizer_conformer}")
    
    # Load model config
    config = OmegaConf.load(config.model.config)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    config.model.train_ds.manifest_filepath = cleaned_train_manifest
    config.model.validation_ds.manifest_filepath = cleaned_test_manifest
    config.model.test_ds.manifest_filepath = cleaned_test_manifest
    
    # Initialize model
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        restore_path="/home/khoatlv/ASR_Nemo/models/conformer/Conformer_small_Model_Language_vi_epoch=250.nemo",
        map_location='cuda'
    )
    
    # Set tokenizer config
    asr_model.cfg.tokenizer.dir = tokenizer_dir
    asr_model.cfg.tokenizer.type = tokenizer_type_cfg

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