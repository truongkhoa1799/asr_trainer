import os
import json
from pathlib import Path
from typing import Dict, List, AnyStr
from scripts.utils import save_manifest, split_train_test_dataset, create_manifest_from_wav_script_dict, config, Logger

Logger = Logger(name="PREPARING_MANIFESTS_INFORE")

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
        if wav_file_name not in list_wave_removed: dict_wave[wav_file_name] = wav_path
    
    Logger.log_info(f"\tTotal script file: {len(list_sctipt_path)} - Actual script file: {len(dict_script.keys())}")
    Logger.log_info(f"\tTotal wav file: {len(list_wave_path)} - Actual wav file: {len(dict_wave.keys())}")
    Logger.log_info("\tDone create dict script and wav")
    
    return dict_wave, dict_script

def prepare_infore_25h(config) -> None:
    Logger.log_info("Start prepare datasets for Infore 25h")
    if config == None:
        Logger.log_error("Fail to get config of Infore 25h")
        exit(-1)
    
    Logger.log_info("\tCreate dict wave and script for Infore 25h")
    dict_wave, dict_script = create_dict_infore_25h(Path(config.wav_dir))
    
    Logger.log_info("\tCreate manifest from dict wave and script for Infore 25h")
    create_manifest_from_wav_script_dict(dict_wave, dict_script, config.manifest)
    
    Logger.log_info("\tCreate Training and Testing Manifest for Infore 25h")
    split_train_test_dataset(config.manifest, config.train_manifest, config.test_manifest, 0.8)
    
    Logger.log_info("Finish prepare datasets for Infore 25h")
    print()
    exit(0)

def prepare_infore_415h(config) -> None:
    manifest_data = []
    number_of_files = 0
    total_duration = 0.0
    with open(config.original_manifest, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            data = json.loads(line)
            
            wave_path = data['key']
            wave_file_name_splited = wave_path.split('/')[6:]
            new_wave_path = os.path.join(config.wav_dir, *wave_file_name_splited)
            
            manifest_data.append(
            {
                "audio_filepath": new_wave_path, 
                "text": data['text'].strip().lower(), 
                "duration": data['duration']
            })
            
            total_duration += float(data["duration"])
            number_of_files += 1
            # if number_of_files >= 100:
            #     break
    
    Logger.log_info("\tCreate Manifest for Infore 415h")
    save_manifest(config.manifest, manifest_data)
    
    Logger.log_info("\tCreate Training and Testing Manifest for Infore 415h")
    split_train_test_dataset(config.manifest, config.train_manifest, config.test_manifest, 0.8)
    
    Logger.log_info(f"\tTotal of file: {number_of_files} - Total Duration: {round(total_duration // 3600, 2)} hours")

if __name__ == "__main__":
    config_infore_25h = config.get_config(["prepare_data", "infore_25h"])
    config_infore_415h = config.get_config(["prepare_data", "infore_415h"])
    
    # prepare_infore_25h(config_infore_25h)
    # prepare_infore_415h(config_infore_415h)
    
    
    
    