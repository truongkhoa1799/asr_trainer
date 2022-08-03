import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, AnyStr
from conformer_asr.utils import save_manifest, read_manifest, split_train_test_dataset, create_manifest_from_wav_script_dict, config, Logger

Logger = Logger(name="PREPARING_MANIFESTS_INFORE_25H")

def create_dict_infore_25h(wave_folder_path: Path) -> List[Dict]:
    dict_wave = dict()
    dict_script = dict()
    
    if wave_folder_path.exists() == False:
        Logger.log_error("\tNot exist infore 25h wav folder")
        return
    
    unaligned_waves = open(wave_folder_path.joinpath('unaligned.txt'), mode='r').readlines()
    list_wave_removed = [unaligned_wave.split('-')[0] for unaligned_wave in unaligned_waves]
    
    list_sctipt_path = list(wave_folder_path.glob("*.txt"))
    list_wave_path = list(wave_folder_path.glob("*.wav"))
    
    # Create Dictionary with key: script_file_name - value: script
    for script_path in list_sctipt_path:
        script = open(script_path, mode='r', encoding='utf-8').readline()
        script = str(script).strip().lower()
        script_file_name, _ = os.path.splitext(script_path)
        script_file_name = script_file_name.split('/')[-1]
        
        if script_file_name not in list_wave_removed and script_file_name != 'unaligned': 
            dict_script[script_file_name] = script

    # Create Dictionary with key: wav_file_name - value: wav_path
    for wav_path in list_wave_path:
        wav_file_name, _ = os.path.splitext(wav_path)
        wav_file_name = wav_file_name.split('/')[-1]
        if wav_file_name not in list_wave_removed: 
            dict_wave[wav_file_name] = str(wav_path)

    Logger.log_info(f"\tTotal script file: {len(list_sctipt_path)} - Actual script file: {len(dict_script.keys())}")
    Logger.log_info(f"\tTotal wav file: {len(list_wave_path)} - Actual wav file: {len(dict_wave.keys())}")
    Logger.log_info("\tDone create dict script and wav")
    
    return dict_wave, dict_script

def create_manifest(config):
    Logger.log_info("Create manifest for Infore 25h")
    if config == None:
        Logger.log_error("Fail to get config of Infore 25h")
        exit(-1)
    
    Logger.log_info("\tCreate dict wave and script for Infore 25h")
    dict_wave, dict_script = create_dict_infore_25h(Path(config.wav_dir))
    
    Logger.log_info("\tCreate manifest from dict wave and script for Infore 25h")
    create_manifest_from_wav_script_dict(dict_wave, dict_script, config.manifest)
    Logger.log_info("Done create manifest for Infore 25h")
    
def split_dataset(config):
    Logger.log_info("Create Training and Testing Manifest for Infore 25h")
    split_train_test_dataset(
        config.manifest, 
        config.train_manifest, 
        config.test_manifest, 0.9
    )
    
    manifest_data, total_duration = read_manifest(config.manifest)
    train_manifest_data, train_duration = read_manifest(config.train_manifest)
    test_manifest_data, test_duration = read_manifest(config.test_manifest)

    Logger.log_info(f"\tNumber of audio: {len(manifest_data)} - Total hours: {round(total_duration/3600, 2)}")
    Logger.log_info(f"\tNumber of training audio: {len(train_manifest_data)} - Total hours: {round(train_duration/3600, 2)}")
    Logger.log_info(f"\tNumber of testing audio: {len(test_manifest_data)} - Total hours: {round(test_duration/3600, 2)}")

    Logger.log_info("Done create Training and Testing Manifest for Infore 25h")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_manifest', action='store_true', help='Create manifest file from original manifest')
    parser.add_argument('--split_dataset', action='store_true', help='Split dataset with cleaned manifest')
    args = parser.parse_args()
    
    config_infore_25h = config.get_config(["prepare_data", "infore_25h"])
    
    if args.create_manifest:
        create_manifest(config_infore_25h)
        
    elif args.split_dataset:
        split_dataset(config_infore_25h)
    
    
    
    