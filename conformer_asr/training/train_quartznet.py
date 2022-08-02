import os
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
from omegaconf import OmegaConf
# Manifest Utils
from tqdm.auto import tqdm
import json
from collections import defaultdict
# Preprocessing steps
import torch.nn as nn
import pytorch_lightning as ptl
import json
from sklearn.model_selection import train_test_split
import re
from datetime import datetime
import copy

# ----------------------------------------------- Parameters ------------------------------------
WANDB_LOGGER = True
ASR_DIR = "/home/khoatlv/ASR-NEMO"
TOKENIZER_DIR = os.path.join(ASR_DIR, "tokenizers")
MANIFEST_DIR = "/home/khoatlv/manifests"

train_manifest = os.path.join(MANIFEST_DIR, "train_manifest.json")
test_manifest = os.path.join(MANIFEST_DIR, "test_manifest.json")

train_manifest_cleaned = "/home/khoatlv/manifests/train_manifest_processed.json"
test_manifest_cleaned = "/home/khoatlv/manifests/test_manifest_processed.json"

model_path = "/home/khoatlv/ASR-NEMO/models/quarznet/stt_en_quartznet15x5.nemo"
model_config = "config/quartznet_15x5.yaml"
config_path = os.path.join(ASR_DIR, model_config)

chars_to_ignore_regex   = '[\,\?\.\!\;\:\"\'\(\)\{\}\“\‘\”\…]|(?:<unk>)'  # remove special character tokens
vocabs = [
    'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x',

    'a', 'á', 'à', 'ạ', 'ã', 'ả',
    'ă', 'ắ', 'ằ', 'ặ', 'ẵ', 'ẳ',
    'â', 'ấ', 'ầ', 'ậ', 'ẫ', 'ẩ',

    'e', 'é', 'è', 'ẹ', 'ẽ', 'ẻ',
    'ê', 'ế', 'ề', 'ệ', 'ễ', 'ể',

    'i', 'í', 'ì', 'ị', 'ĩ', 'ỉ',
    'y', 'ý', 'ỳ', 'ỵ', 'ỹ', 'ỷ',
    
    'o', 'ó', 'ò', 'ọ', 'õ', 'ỏ',
    'ô', 'ố', 'ồ', 'ộ', 'ỗ', 'ổ',
    'ơ', 'ớ', 'ờ', 'ợ', 'ỡ', 'ở',

    'u', 'ú', 'ù', 'ụ', 'ũ', 'ủ',
    'ư', 'ứ', 'ừ', 'ự', 'ữ', 'ử',
    
    'j', 'f', 'w', 'z', ' '
]
vocabs_file = os.path.join(ASR_DIR, "vocabs.txt")

# Parameters for creating manifest
use_common_voice = True
use_vivos = True
use_vlsp_set_2 = True

use_fpt = False
use_vlsp_set_1 = False

common_voice_manifest_dir = "/home/khoatlv/data/common_voice/manifests"
vivos_manifest_dir = "/home/khoatlv/data/vivos/manifests"
vlsp_manifest_dir = "/home/khoatlv/data/vlsp2020/manifests"

common_voice_test_manifest = ""
common_voice_dev_manifest = ""
common_voice_train_manifest = ""
vivos_test_manifest = ""
vivos_train_manifest = ""
vlsp2020_train_set_01_manifest = ""
vlsp2020_test_set_01_manifest = ""
vlsp2020_train_set_02_manifest = ""
vlsp2020_test_set_02_manifest = ""

vocabs_set = set(vocabs)
train_set = None
test_set = None

## -------------------------------------------------- function needed-------------------------------------

def create_manifest():
    global common_voice_test_manifest, common_voice_dev_manifest, common_voice_train_manifest
    global vivos_test_manifest, vivos_train_manifest
    global vlsp2020_train_set_01_manifest, vlsp2020_test_set_01_manifest
    global vlsp2020_train_set_02_manifest, vlsp2020_test_set_02_manifest

    # Common_voice dataset
    if use_common_voice:
        print("Create manifest for common voice dataset")
        common_voice_test_manifest = os.path.join(common_voice_manifest_dir, "commonvoice_test_manifest.json")
        common_voice_dev_manifest = os.path.join(common_voice_manifest_dir, "commonvoice_dev_manifest.json")
        common_voice_train_manifest = os.path.join(common_voice_manifest_dir, "commonvoice_train_manifest.json")

    # Vivos dataset
    if use_vivos:
        print("Create manifest for vivos dataset")
        vivos_test_manifest = os.path.join(vivos_manifest_dir, "test_manifest.json")
        vivos_train_manifest = os.path.join(vivos_manifest_dir, "train_manifest.json")

    # # FPT dataset
    # if use_fpt:
    #     fpt_manifest_dir = "/home/khoatlv/data/FPT/manifests"
    #     fpt_test_manifest = os.path.join(fpt_manifest_dir, "FPT_test_manifest.json")
    #     fpt_train_manifest = os.path.join(fpt_manifest_dir, "FPT_train_manifest.json")
    # else:
    #     fpt_test_manifest = ""
    #     fpt_train_manifest = ""

    # vlsp dataset
    if use_vlsp_set_1:
        print("Create manifest for vlsp set 1 dataset")
        manifest_data = []
        vlsp_set_path = os.path.join(vlsp_manifest_dir, "vlsp2020_set_01_manifest.json")

        vlsp2020_train_set_01_manifest = os.path.join(vlsp_manifest_dir, "vlsp2020_train_set_01_manifest.json")
        vlsp2020_test_set_01_manifest = os.path.join(vlsp_manifest_dir, "vlsp2020_test_set_01_manifest.json")

        with open(vlsp_set_path, 'r') as f:
            for line in tqdm(f, desc="Reading manifest data vlsp set 1"):
                line = line.replace("\n", "")
                data = json.loads(line)
                manifest_data.append(data)

        train_dataset, test_dataset = train_test_split(manifest_data, train_size=0.9, test_size=0.1, random_state=42)
        print("Split {} data from vlsp2020 set 1 to {} train and {} test".format(len(manifest_data), len(train_dataset), len(test_dataset)))

        with open(vlsp2020_train_set_01_manifest, mode='w', encoding='utf-8') as f:
            data = '\n'.join([json.dumps(i, ensure_ascii=False) for i in train_dataset])
            f.write(data + '\n')
        
        with open(vlsp2020_test_set_01_manifest, mode='w', encoding='utf-8') as f:
            data = '\n'.join([json.dumps(i, ensure_ascii=False) for i in test_dataset])
            f.write(data + '\n')

    if use_vlsp_set_2:
        print("Create manifest for vlsp set 2 dataset")
        manifest_data = []
        vlsp_set_path = os.path.join(vlsp_manifest_dir, "vlsp2020_set_02_manifest.json")

        vlsp2020_train_set_02_manifest = os.path.join(vlsp_manifest_dir, "vlsp2020_train_set_02_manifest.json")
        vlsp2020_test_set_02_manifest = os.path.join(vlsp_manifest_dir, "vlsp2020_test_set_02_manifest.json")

        with open(vlsp_set_path, 'r') as f:
            for line in tqdm(f, desc="Reading manifest data vlsp set 2"):
                line = line.replace("\n", "")
                data = json.loads(line)
                manifest_data.append(data)

        train_dataset, test_dataset = train_test_split(manifest_data, train_size=0.9, test_size=0.1, random_state=42)
        print("Split {} data from vlsp2020 set 2 to {} train and {} test".format(len(manifest_data), len(train_dataset), len(test_dataset)))

        with open(vlsp2020_train_set_02_manifest, mode='w', encoding='utf-8') as f:
            data = '\n'.join([json.dumps(i, ensure_ascii=False) for i in train_dataset])
            f.write(data + '\n')
        
        with open(vlsp2020_test_set_02_manifest, mode='w', encoding='utf-8') as f:
            data = '\n'.join([json.dumps(i, ensure_ascii=False) for i in test_dataset])
            f.write(data + '\n')

    if os.path.exists(test_manifest): os.remove(test_manifest)
    if os.path.exists(train_manifest): os.remove(train_manifest)
    
    os.system("cat {common_voice_train_manifest} {common_voice_dev_manifest} {vivos_train_manifest} {vlsp2020_train_set_01_manifest} {vlsp2020_train_set_02_manifest} > {train_manifest}".format(
        common_voice_train_manifest=common_voice_train_manifest,
        common_voice_dev_manifest=common_voice_dev_manifest,
        vivos_train_manifest=vivos_train_manifest,
        vlsp2020_train_set_01_manifest=vlsp2020_train_set_01_manifest,
        vlsp2020_train_set_02_manifest=vlsp2020_train_set_02_manifest,
        train_manifest=train_manifest
    ))

    os.system("cat {common_voice_test_manifest} {vivos_test_manifest} {vlsp2020_test_set_01_manifest} {vlsp2020_test_set_02_manifest} > {test_manifest}".format(
        common_voice_test_manifest=common_voice_test_manifest,
        vivos_test_manifest=vivos_test_manifest,
        vlsp2020_test_set_01_manifest=vlsp2020_test_set_01_manifest,
        vlsp2020_test_set_02_manifest=vlsp2020_test_set_02_manifest,
        test_manifest=test_manifest
    ))
    print()


def read_manifest(path):
        manifest = []
        with open(path, 'r') as f:
            for line in tqdm(f, desc="Reading manifest data"):
                line = line.replace("\n", "")
                data = json.loads(line)
                manifest.append(data)
        return manifest

def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum, ensure_ascii=False)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath

def get_charset(manifest_data):
    global vocabs
    charset = defaultdict(int)
    invalid_token = []
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text'].strip().lower()
        for character in text:
            if character not in vocabs:
                invalid = {
                    "token": character,
                    "data": row
                }
                invalid_token.append(invalid)
            else:
                charset[character] += 1
    return charset, invalid_token


def remove_special_characters(data):
    data["text"] = re.sub(chars_to_ignore_regex, '', data["text"]).lower().strip()
    return data

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest

def preprocessing_data():
    global train_manifest_cleaned, test_manifest_cleaned, vocabs_set, vocabs
    with open(vocabs_file, mode='w') as fout:
        fout.write(str(vocabs))

    # List of pre-processing functions
    PREPROCESSORS = [
        remove_special_characters,
    ]

    # Load manifests
    train_data = read_manifest(train_manifest)
    test_data = read_manifest(test_manifest)

    # Apply preprocessing
    train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
    test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)

    # Write new manifests
    train_manifest_cleaned = write_processed_manifest(train_data_processed, train_manifest)
    test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest)
    print()


def extract_character_set():
    global train_set, test_set
    train_manifest_data = read_manifest(train_manifest_cleaned)
    test_manifest_data = read_manifest(test_manifest_cleaned)

    train_charset, invalid_token_train = get_charset(train_manifest_data)
    test_charset, invalid_token_test = get_charset(test_manifest_data)

    train_set = set(train_charset.keys())
    test_set = set(test_charset.keys())

    print(f"Number of tokens in vocabs set : {len(vocabs_set)}")
    print(vocabs_set)
    print()

    print(f"Number of tokens in preprocessed train set : {len(train_set)}")
    print(train_set)
    print(f"Number of tokens in preprocessed test set : {len(test_set)}")
    print(test_set)
    print()

    intersection_train_test = set.intersection(train_set, test_set)
    train_oov = train_set - intersection_train_test
    test_oov = test_set - intersection_train_test

    print(f"Number of OOV tokens in train set : {len(train_oov)}")
    print(train_oov)
    print(f"Number of OOV tokens in test set : {len(test_oov)}")
    print(test_oov)
    print()

    intersection_train_vocabs = set.intersection(train_set, vocabs_set)
    train_oov_vocabs = vocabs_set - intersection_train_vocabs
    print(f"Number of OOV tokens in train set with vocabs : {len(train_oov_vocabs)}")
    print(train_oov_vocabs)

    intersection_test_vocabs = set.intersection(test_set, vocabs_set)
    test_oov_vocabs = vocabs_set - intersection_test_vocabs
    print(f"Number of OOV tokens in test set with vocabs : {len(test_oov_vocabs)}")
    print(test_oov_vocabs)

    invalid_token_train_log = os.path.join("tokenizers", "invalid_token_train.txt")
    invalid_token_test_log = os.path.join("tokenizers", "invalid_token_test.txt")

    with open(invalid_token_test_log, mode='w') as fout_test:
        data = '\n'.join([json.dumps(i, ensure_ascii=False) for i in invalid_token_test])
        fout_test.write(data + '\n')

    with open(invalid_token_train_log, mode='w') as fout_train:
        data = '\n'.join([json.dumps(i, ensure_ascii=False) for i in invalid_token_train])
        fout_train.write(data + '\n')


def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

def load_config(path):
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    config.model.train_ds.manifest_filepath = train_manifest_cleaned
    config.model.validation_ds.manifest_filepath = test_manifest_cleaned
    config.model.test_ds.manifest_filepath = test_manifest_cleaned
    
    return config


if __name__ == '__main__':
    # wandb login
    os.system("wandb login {}".format("03f1412a8edbcb2869809c69eb534d8b803365b2"))
      
    create_manifest()
    preprocessing_data()
    extract_character_set()

    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(
        model_path,
        map_location='cuda'
    )
    # asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(
    #     "/home/khoatlv/ASR-NEMO/experiments/ASR-Char-Model-Language-vi/2022-03-24_10-28-21/checkpoints/ASR-Char-Model-Language-vi--val_wer=0.3729-epoch=8-last.ckpt",
    #     map_location='cuda'
    # )
    
    freeze_encoder = False 
    freeze_encoder = bool(freeze_encoder)

    if freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        print("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        asr_model.encoder.unfreeze()
        print("Model encoder has been un-frozen")

    config = load_config(config_path)
    asr_model.change_vocabulary(new_vocabulary=list(config.model.labels))
    asr_model.cfg.labels = list(config.model.labels)
    
    asr_model.setup_training_data(config.model.train_ds)
    asr_model.setup_multiple_validation_data(config.model.validation_ds)

    with open_dict(asr_model.cfg):
        asr_model.cfg.optim = config.model.optim
        asr_model.cfg.spec_augment = config.model.spec_augment    
        
    asr_model.setup_optimization(config.model.optim)
    asr_model.spec_augmentation = asr_model.from_config_dict(config.model.spec_augment)

    asr_model._wer.use_cer = True
    asr_model._wer.log_prediction = True

    trainer = ptl.Trainer(**config.trainer)
    asr_model.set_trainer(trainer)
    asr_model.cfg = asr_model._cfg

    exp_config = exp_manager.ExpManagerConfig(**config.exp_manager)
    exp_config = OmegaConf.structured(exp_config)
    logdir = exp_manager.exp_manager(trainer, exp_config)

    print("Starting train model")
    trainer.fit(asr_model)

    print("Save model")
    save_path = "/home/khoatlv/ASR-NEMO/models/quarznet/quartznet_15x5_trained.nemo"
    asr_model.save_to(save_path)