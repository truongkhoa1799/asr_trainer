import json
import librosa
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import logging
import sys
import os
import argparse

def read_manifest(manifest_path, ignore_data=False):
    manifest = []
    total_duration = 0
    
    if manifest_path is None: return manifest, total_duration
    with open(manifest_path, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            data = json.loads(line)
            total_duration += data["duration"]
            
            if not ignore_data:
                manifest.append(data)
                
    return manifest, total_duration

def save_manifest(manifest_path, list_record):
    with open(manifest_path, mode='w', encoding='utf-8') as fout:
        data = '\n'.join(json.dumps(i, ensure_ascii=False) for i in list_record)
        fout.writelines(data + '\n')
        fout.close()
        
def split_train_test_dataset(manifest_path, out_train_manifest, out_test_manifest, train_per):
    list_json_data, _ = read_manifest(manifest_path)
    train_dataset, test_dataset = train_test_split(list_json_data, train_size=train_per, test_size=1-train_per, random_state=42)
    save_manifest(out_train_manifest, train_dataset)
    save_manifest(out_test_manifest, test_dataset)
    
def create_manifest_from_wav_script_dict(dict_wav, dict_script, manifest_path):
    list_manifest = list()
    for wav_name, wav_path in dict_wav.items():
        duration = librosa.get_duration(filename=wav_path)
        list_manifest.append(
            {
                "audio_filepath": wav_path, 
                "text": dict_script[wav_name], 
                "duration": round(duration, 2)
            }
        )
    
    save_manifest(manifest_path, list_manifest)
    
def summarize_dataset(config):
    LOGGER = Logger(name="SUMMERIZE_DATASETS")
    LOGGER.log_info("Summerize dataset")
    prepare_data = config.get_config(["prepare_data"])
    config = config.get_config(["training"])
    
    datasets = prepare_data.keys()
    for dataset in datasets:
        if not config["use_datasets"][f"use_{dataset}"]: continue
        
        if dataset == "common_voice":
            train_manifest = prepare_data['common_voice']['train_manifest']
            dev_manifest = prepare_data['common_voice']['dev_manifest']
            test_manifest = prepare_data['common_voice']['test_manifest']
        else:
            train_manifest = prepare_data[dataset].train_manifest
            dev_manifest = None
            test_manifest = prepare_data[dataset].test_manifest
        
        num_train_data = os.popen(f"wc -l {train_manifest}").read().strip()
        num_test_data = os.popen(f"wc -l {test_manifest}").read().strip()
        _, total_duration_train = read_manifest(train_manifest, ignore_data=True)
        _, total_duration_test = read_manifest(test_manifest, ignore_data=True)
        
        if dev_manifest is not None:
            num_dev_data = os.popen(f"wc -l {dev_manifest}").read().strip()
            _, total_duration_dev = read_manifest(dev_manifest, ignore_data=True)
            
        total_hours = round((total_duration_train+total_duration_test+total_duration_dev)/3600, 2)
        LOGGER.log_info(f"Dataset {dataset} with total hours: {total_hours}")
        LOGGER.log_info(f"\tTraining data {num_train_data}, total hours {round(total_duration_train/3600, 2)}")
        LOGGER.log_info(f"\tTesting data {num_test_data}, total hours {round(total_duration_test/3600, 2)}")
        if dev_manifest is not None:
            LOGGER.log_info(f"\tDev data {num_dev_data}, total hours {round(total_duration_dev/3600, 2)}")
        
        print()

class Config():
    def __init__(self, config_path) -> None:
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        return config

    def get_config(self, keys=None):
        if keys is None: return self.config
        config = self.config
        for key in keys:
            if key in config:
                config = config[key]
            else: return None
        
        return config

class Logger():
    def __init__(self, name) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        
        # file_handler = logging.FileHandler("/home/khoatlv/Conformer_ASR/log/trainig.txt")
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)
        
        # logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)
    
    def log_info(self, message=""):
        logging.disable(logging.NOTSET)
        self.logger.info(message)
        logging.disable(logging.INFO)
    
    def log_error(self, message=""):
        self.logger.error(message)

config = Config("/home/khoatlv/ASR_Nemo/config/training_config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--summarize_dataset',
        action='store_true',
        help='Summerize dataset information'
    )
    args = parser.parse_args()
    
    if args.summarize_dataset:
        summarize_dataset(config)