import os
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import exp_manager
import nemo
from omegaconf import OmegaConf
# Manifest Utils
from tqdm.auto import tqdm
import json
# Preprocessing steps
import torch.nn as nn
import pytorch_lightning as ptl
import json
from datetime import datetime
from collections import defaultdict
import copy
import torch

WANDB_LOGGER = True
ASR_DIR = "/home/khoatlv/ASR-NEMO"

VOCAB_SIZE = 512
tokenizer_dir = os.path.join(ASR_DIR, "tokenizers_citrinet")
TOKENIZER_TYPE = "spe"
TOKENIZER_DIR = f"{tokenizer_dir}/tokenizer_spe_{TOKENIZER_TYPE}_v{VOCAB_SIZE}/"

# Tokenizer path
if TOKENIZER_TYPE == 'spe':
  TOKENIZER_TYPE_CFG = "bpe"
  TOKENIZER_DIR = os.path.join(tokenizer_dir, f"tokenizer_spe_{TOKENIZER_TYPE_CFG}_v{VOCAB_SIZE}")
else:
  TOKENIZER_TYPE_CFG = "wpe"
  TOKENIZER_DIR = os.path.join(tokenizer_dir, f"tokenizer_wpe_v{VOCAB_SIZE}")

print("Tokenizer directory :", TOKENIZER_DIR)

model_config = "config/citrinet_256.yaml"
config_path = os.path.join(ASR_DIR, model_config)

train_manifest_cleaned = "/home/khoatlv/manifests/train_manifest_processed.json"
test_manifest_cleaned = "/home/khoatlv/manifests/test_manifest_processed.json"

def load_config(path):
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    config.model.train_ds.manifest_filepath = train_manifest_cleaned
    config.model.validation_ds.manifest_filepath = test_manifest_cleaned
    config.model.test_ds.manifest_filepath = test_manifest_cleaned
    
    return config

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

if __name__ == "__main__":
    os.system("wandb login {}".format("03f1412a8edbcb2869809c69eb534d8b803365b2"))

    asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_citrinet_256", map_location='cuda')
    asr_model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR, new_tokenizer_type="bpe")

    # asr_model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(
    #     "/home/khoatlv/ASR-NEMO/experiments/Citrinet-256-Model-Language-vi/2022-03-28_10-53-12/checkpoints/Citrinet-256-Model-Language-vi--val_wer=0.3136-epoch=4-last.ckpt", 
    #     map_location='cuda'
    # )
    config = load_config(model_config)

    freeze_encoder = False
    if freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        print("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        asr_model.encoder.unfreeze()
        print("Model encoder has been un-frozen")

    # Set tokenizer config
    asr_model.cfg.tokenizer.dir = TOKENIZER_DIR
    asr_model.cfg.tokenizer.type = TOKENIZER_TYPE_CFG

    asr_model.setup_training_data(config.model.train_ds)
    asr_model.setup_multiple_test_data(config.model.test_ds)
    asr_model.setup_multiple_validation_data(config.model.validation_ds)

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

    # Train the model
    trainer.fit(asr_model)