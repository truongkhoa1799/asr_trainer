#!/usr/bin/env python3
import logging
import os
import sys
from nemo.collections.asr.models import EncDecCTCModelBPE
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as ptl
from nemo.utils import exp_manager
import torch.nn as nn
from conformer_asr.utils import config, Logger, Config

WANDB_LOGGER = True

# -------------------------------- FUNCTIONS --------------------------------
def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

# -------------------------------- MAIN --------------------------------
if __name__ == "__main__":
    training_config = config.get_config(["training"])
    
    # LOAD CONFIG
    model_config = Config(training_config.model.config).get_config()
    model_config.model.train_ds.manifest_filepath = training_config.manifest.train_manifest_cleaned
    model_config.model.validation_ds.manifest_filepath = training_config.manifest.test_manifest_cleaned
    logging.info("\tLoad model config and change dataset")
    
    # CONFIG WANDB
    if training_config.wandb.is_use:
        logging.info("Use Wandb for logging result")
        os.system(f"wandb login {training_config.wandb.key}")
    
    # CONFIG TOKENIZER
    tokenizer_dir = training_config.tokenizer.tokenizer_dir
    tokenizer_type = training_config.tokenizer.type
    tokenizer_type_cfg = training_config.tokenizer.type_cfg
    vocab_size = config.get_config(["training", "vocab_size"])
    tokenizer_conformer_dir = os.path.join(tokenizer_dir, f"tokenizer_{tokenizer_type}_{tokenizer_type_cfg}_v{vocab_size}")
    
    logging.info(f"\tUse Tokenizer Type {tokenizer_type}")
    logging.info(f"\tUse Tokenizer Type Config {tokenizer_type_cfg}")
    logging.info(f"\tTokenizer Dir {tokenizer_conformer_dir}")
    
    if training_config.model.use_pretrained:
        logging.info(f"\tUse pretrained {training_config.model.from_pretrained} \
            with tokenizer dir{tokenizer_conformer_dir} and tokenizer type config {tokenizer_type_cfg}")
        
        asr_model = EncDecCTCModelBPE.from_pretrained(
            model_name=training_config.model.from_pretrained,
            map_location=training_config.model.device
        )
        asr_model.change_vocabulary(
            new_tokenizer_dir=tokenizer_conformer_dir, 
            new_tokenizer_type=tokenizer_type_cfg
        )
    else:
        logging.info(f"\tFine tune Conformer Model based on {training_config.model.finetuned_model}")
        asr_model = EncDecCTCModelBPE.restore_from(
            restore_path=training_config.model.finetuned_model,
            map_location=training_config.model.device
        )
    
    freeze_encoder = bool(training_config.model.freeze_encoder)
    if freeze_encoder:
        logging.info("\tModel encoder has been frozen, and batch normalization has been unfrozen")
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
    else:
        asr_model.encoder.unfreeze()
        logging.info("\tModel encoder has been un-frozen")

    # Set tokenizer config
    logging.info("\tUpdate Tokenizer Config for ASR Model")
    asr_model.cfg.tokenizer.dir = tokenizer_conformer_dir
    asr_model.cfg.tokenizer.type = tokenizer_type_cfg

    logging.info("\tUpdate Training and Testing dataset path for ASR Model")
    asr_model.setup_training_data(model_config.model.train_ds)
    asr_model.setup_validation_data(model_config.model.validation_ds)
    asr_model.setup_multiple_test_data(model_config.model.test_ds)

    logging.info("\tUpdate Opmizer and Spec_Argument for ASR Model")
    with open_dict(asr_model.cfg):
        asr_model.cfg.optim = model_config.model.optim
        asr_model.cfg.spec_augment = model_config.model.spec_augment    
    asr_model.spec_augmentation = asr_model.from_config_dict(model_config.model.spec_augment)
    asr_model.setup_optimization(model_config.model.optim)

    asr_model._wer.use_cer = True
    asr_model._wer.log_prediction = True

    logging.info("\tCreate ASR Model Trainer")
    trainer = ptl.Trainer(**model_config.trainer)
    asr_model.set_trainer(trainer)
    asr_model.cfg = asr_model._cfg

    logging.info("\tCreate ASR Model Experiment Manager")
    exp_config = exp_manager.ExpManagerConfig(**model_config.exp_manager)
    exp_config = OmegaConf.structured(exp_config)
    logdir = exp_manager.exp_manager(trainer, exp_config)

    # Train the model
    trainer.fit(asr_model)