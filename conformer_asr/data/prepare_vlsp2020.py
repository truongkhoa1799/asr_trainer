import os
import sys
import copy
import glob
import librosa
import argparse
from enum import Enum
from pathlib import Path
import soundfile as sf
from typing import TypedDict, List
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from conformer_asr.utils import Logger, save_manifest, read_manifest, config, split_train_test_dataset, ManifestData

class TrainSet(Enum):
    Set_1 = 0
    Set_2 = 1

class DatasetConfig(TypedDict):
    dataset_name: str
    list_script_path: List[str]
    list_audio_path: List[str]
    config

LOGGER = Logger(name="PREPARING_MANIFESTS_VLSP_2020")

def validate_sample_rate(audio_path):
    y, rate = sf.read(audio_path)
    return "0" if rate==16000 else audio_path

def create_transcript_dict(list_script_path):
    transcipt_dict = dict()
    for idx, script_path in enumerate(list_script_path):
        with open(script_path, mode='r') as f:
            script = f.readline().strip()
            transcipt_dict[os.path.splitext(script_path)[0]] = script
            f.close()
        
        if idx % 10000 == 0: LOGGER.log_info(f"\t\tRead {idx} script files")
    return transcipt_dict

def create_train_manifest(config):
    num_unmap = 0
    invalid_files = []
    training_data = []
    invalid_data = []
    if os.path.exists(config['config'].train_manifest): os.remove(config['config'].train_manifest)
    
    LOGGER.log_info(f"Start create training manifest for {config['dataset_name']}")
    LOGGER.log_info(f"\tNumber of scripts {len(config['list_script_path'])}")
    LOGGER.log_info(f"\tNumber of waves {len(config['list_audio_path'])}")
    
    # Create dictionary transcipt
    LOGGER.log_info(f"\tCreate transcript dict")
    transcipt_dict = create_transcript_dict(list_script_path=config['list_script_path'])
    LOGGER.log_info(f"\tDone create transcript dict")
    
    # Map audio and transcript to create manifest
    LOGGER.log_info("\tWrite audio and transcript to manifest")
    for idx, audio_path in enumerate(config['list_audio_path']):
        audio_name, ext = os.path.splitext(audio_path)
        if audio_name not in transcipt_dict:
            num_unmap += 1
            continue
    
        duration = librosa.get_duration(filename=str(audio_path))
        data = ManifestData(
            audio_filepath=str(audio_path),
            duration=duration,
            text=transcipt_dict[audio_name]
        )
         
        if "<unk>" in transcipt_dict[audio_name]: invalid_data.append(data)
        else: training_data.append(data)
        
        if idx % 10000 == 0: LOGGER.log_info(f"\t\tWrite {idx} script files")
    LOGGER.log_info("\tDone write audio and transcript to manifest")
                   
    save_manifest(config['config'].train_manifest, training_data)
    save_manifest(config['config'].invalid_manifest, invalid_data)
    
    LOGGER.log_info(f"\tNumber of training audio: {len(training_data)}")
    LOGGER.log_info(f"\tNumber of unmapped audio and transcipt: {num_unmap}")
    LOGGER.log_info("\tInvalid dataset {}".format(len(invalid_data)))
    LOGGER.log_info(f"Done create training manifest for {config['dataset_name']}")

def create_test_manifest(config):
    num_unmap = 0
    invalid_files = []
    testing_data = []
    invalid_data = []
    if os.path.exists(config['config'].test_manifest): os.remove(config['config'].test_manifest)
    
    transcipt_dict = {}
    with open(str(config['list_script_path'])) as fin:
        for script_line in fin.readlines():
            script_line = script_line.strip().split(" ")
            audio_name = script_line[0]
            script = " ".join(script_line[5:])
            transcipt_dict[audio_name] = script
                
    LOGGER.log_info(f"Start create testing manifest for {config['dataset_name']}")
    LOGGER.log_info(f"\tNumber of scripts {len(transcipt_dict.items())}")
    LOGGER.log_info(f"\tNumber of waves {len(config['list_audio_path'])}")
    
    # Map audio and transcript to create manifest
    LOGGER.log_info("\tWrite audio and transcript to manifest")
    for idx, audio_path in enumerate(config['list_audio_path']):
        audio_name, ext = os.path.splitext(audio_path)
        audio_name = audio_name.split("/")[-1]
    
        if audio_name not in transcipt_dict:
            num_unmap += 1
            continue
    
        duration = librosa.get_duration(filename=str(audio_path))
        data = ManifestData(
            audio_filepath=str(audio_path),
            duration=duration,
            text=transcipt_dict[audio_name]
        )
         
        if "<unk>" in transcipt_dict[audio_name]: invalid_data.append(data)
        else: testing_data.append(data)
        
        if idx % 10000 == 0: LOGGER.log_info(f"\t\tWrite {idx} script files")
    LOGGER.log_info("\tDone write audio and transcript to manifest")
                   
    save_manifest(config['config'].test_manifest, testing_data)
    
    LOGGER.log_info(f"\tNumber of testing audio: {len(testing_data)}")
    LOGGER.log_info(f"\tNumber of unmapped audio and transcipt: {num_unmap}")
    LOGGER.log_info("\tInvalid dataset {}".format(len(invalid_data)))
    LOGGER.log_info(f"Done create testing manifest for {config['dataset_name']}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_1', action='store_true', help='Split dataset with cleaned manifest')
    parser.add_argument('--set_2', action='store_true', help='Split dataset with cleaned manifest')
    args = parser.parse_args()
    
    if args.set_1:
        config = config.get_config(["prepare_data", "vlsp2020_set1"])
        training_dataset_config = DatasetConfig(
            config = config,
            dataset_name = "VLSP-2020 set 1",
            list_script_path = list(Path(config.train_data_dir).glob("*/*.txt")),
            list_audio_path = list(Path(config.train_data_dir).glob("*/*.wav"))
        )
        create_train_manifest(training_dataset_config)
        
        testing_dataset_config = DatasetConfig(
            config = config,
            dataset_name = "VLSP-2020 set 1",
            list_script_path = "/home/khoatlv/data/vlsp2020/vlsp2020_test_set_01/task-01.stm.out",
            list_audio_path = list(Path(config.test_data_dir).glob("*.wav"))
        )
        create_test_manifest(testing_dataset_config)
        
    elif args.set_2:
        config = config.get_config(["prepare_data", "vlsp2020_set2"])
        training_dataset_config = DatasetConfig(
            config = config,
            dataset_name = "VLSP-2020 set 2",
            list_script_path = glob.glob(os.path.join(config.train_data_dir, "*.txt")),
            list_audio_path = glob.glob(os.path.join(config.train_data_dir, "*.wav"))
        )
        create_train_manifest(training_dataset_config)
        
        testing_dataset_config = DatasetConfig(
            config = config,
            dataset_name = "VLSP-2020 set 2",
            list_script_path = "/home/khoatlv/data/vlsp2020/vlsp2020_test_set_02/task-02.stm.out",
            list_audio_path = glob.glob(os.path.join(config.test_data_dir, "*.wav"))
        )
        create_test_manifest(testing_dataset_config)
