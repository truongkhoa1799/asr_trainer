import os
import sys
import copy
import glob
import librosa
from enum import Enum
import soundfile as sf
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

PROJECT_DIR = "/".join(os.getcwd().split('/')[:-1])
sys.path.append(PROJECT_DIR)
from Conformer_ASR.scripts.utils import Logger, save_manifest, config

class TrainSet(Enum):
    Set_1 = 0
    Set_2 = 1

TRAIN_PER = 0.9
TEST_PER = 0.1
ROW_FORMAT = {"audio_filepath": "", "duration": 0, "text": ""}
LOGGER = Logger(name="PREPARING_MANIFESTS_VLSP")

def validate_sample_rate(audio_path):
    y, rate = sf.read(audio_path)
    return "0" if rate==16000 else audio_path

def create_transcript_dict(text_script):
    transcipt_dict = dict()
    for idx, file in enumerate(text_script):
        with open(file, mode='r') as f:
            text = f.readline().strip()
            transcipt_dict[os.path.splitext(file)[0]] = text
            f.close()
        
        if idx % 10000 == 0: LOGGER.log_info(f"\t\tRead {idx} script files")
    return transcipt_dict

def create_manifest_function(set_num, config):
    num_unmap = 0
    invalid_files = []
    
    all_data = []
    invalid_data = []
    
    dataset_name = ""
    text_script = None
    audio_path = None

    if set_num == TrainSet.Set_1:
        dataset_name = "VLSP-2020 set 1"
        text_script = glob.glob(os.path.join(config.data_dir, "*/*.txt"))
        audio_path = glob.glob(os.path.join(config.data_dir, "*/*.wav"))
        
    elif set_num == TrainSet.Set_2:
        dataset_name = "VLSP-2020 set 2"
        text_script = glob.glob(os.path.join(config.data_dir, "*.txt"))
        audio_path = glob.glob(os.path.join(config.data_dir, "*.wav"))

    LOGGER.log_info(f"Start prepare datasets for {dataset_name}")
    LOGGER.log_info(f"\tNumber of scripts {len(text_script)}")
    LOGGER.log_info(f"\tNumber of waves {len(audio_path)}")
    LOGGER.log_info(f"\tCreate transcript dict")
    transcipt_dict = create_transcript_dict(text_script=text_script)
    
    LOGGER.log_info("\tValidate sample rate of wav")
    with Parallel(n_jobs=24) as parallel:
        invalid_files = parallel(delayed(validate_sample_rate)(audio_path) for audio_path in audio_path)
    
    num_invalid = invalid_files.count(invalid_files != "0")
    if num_invalid != 0:
        with Parallel(n_jobs=24) as parallel:
            invalid_files = parallel(
                delayed(validate_sample_rate)(audio_path) 
                for audio_path in audio_path
            )
        
    LOGGER.log_info(f"\tNumber of invalid waves: {num_invalid}")
    if num_invalid != 0: return
    if os.path.exists(config.manifest): os.remove(config.manifest)

    # Map audio and transcript to create manifest
    LOGGER.log_info("\tWrite audio and transcript to manifest")
    for idx, path in enumerate(audio_path):
        audio_name, ext = os.path.splitext(path)
        if audio_name not in transcipt_dict:
            num_unmap += 1
            continue
            
        text = transcipt_dict[audio_name]
        if "<unk>" in text: invalid_data.append(data)
        else:
            data, sr = sf.read(path)
            duration = librosa.get_duration(y=data, sr=sr)
            data = copy.deepcopy(ROW_FORMAT)
            data["audio_filepath"] = path
            data["duration"] = duration
            data["text"] = text
            all_data.append(data)
        
        if idx % 10000 == 0: LOGGER.log_info(f"\t\tWrite {idx} script files")
                    
    # train_dataset, test_dataset = train_test_split(all_data, train_size=TRAIN_PER, test_size=TEST_PER, random_state=42)
    # save_manifest(config.manifest, all_data)
    # save_manifest(config.train_manifest, train_dataset)
    # save_manifest(config.test_manifest, test_dataset)
    # save_manifest(config.invalid_manifest, invalid_data)
    
    save_manifest(config.train_manifest, all_data)
    
    LOGGER.log_info(f"\tNumber of unmapped audio and transcipt: {num_unmap}")
    LOGGER.log_info("\tInvalid dataset {}".format(len(invalid_data)))
    LOGGER.log_info("\tSplit {} data from {} to {} train and {} test".format(len(all_data), dataset_name, len(train_dataset), len(test_dataset)))
    LOGGER.log_info(f"Finish prepare datasets for {dataset_name}")
if __name__ == '__main__':
    # config_set_1 = config.get_config(["prepare_data", "vlsp2020_set1"])
    # create_manifest_function(TrainSet.Set_1, config_set_1)
    # print()
    
    config_set_2 = config.get_config(["prepare_data", "vlsp2020_set2"])
    create_manifest_function(TrainSet.Set_2, config_set_2)
    print()

