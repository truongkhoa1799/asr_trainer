import os
import argparse
from conformer_asr.utils import save_manifest, split_train_test_dataset, config, Logger, read_manifest

Logger = Logger(name="PREPARING_MANIFESTS_INFORE_415H")

def create_manifest(config):
    Logger.log_info("Create manifest for Infore 415h from original manifest")
    manifest_data = []
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
    
    save_manifest(config.manifest, manifest_data)
    Logger.log_info("Done create manifest for Infore 415h from original manifest")

def split_dataset(config):
    Logger.log_info("Create Training and Testing Manifest for Infore 415h")
    split_train_test_dataset(
        config.manifest_cleaned, 
        config.train_manifest, 
        config.test_manifest, 
        0.9
    )
    cleaned_manifest_data, total_duration = read_manifest(config.manifest_cleaned)
    train_manifest_data, train_duration = read_manifest(config.train_manifest)
    test_manifest_data, test_duration = read_manifest(config.test_manifest)

    Logger.log_info(f"\tNumber of audio: {len(cleaned_manifest_data)} - Total hours: {round(total_duration/3600, 2)}")
    Logger.log_info(f"\tNumber of training audio: {len(train_manifest_data)} - Total hours: {round(train_duration/3600, 2)}")
    Logger.log_info(f"\tNumber of testing audio: {len(test_manifest_data)} - Total hours: {round(test_duration/3600, 2)}")
    Logger.log_info("Done create Training and Testing Manifest for Infore 415h")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_manifest', action='store_true', help='Create manifest file from original manifest')
    parser.add_argument('--split_dataset', action='store_true', help='Split dataset with cleaned manifest')
    args = parser.parse_args()
    
    config_infore_415h = config.get_config(["prepare_data", "infore_415h"])
    
    if args.create_manifest:
        create_manifest(config_infore_415h)
        
    elif args.split_dataset:
        split_dataset(config_infore_415h)
        
        
    