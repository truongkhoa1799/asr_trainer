import os
import sys
import json
from prometheus_client import Enum
from sklearn.model_selection import train_test_split

PROJECT_DIR = "/".join(os.getcwd().split('/')[:-1])
sys.path.append(PROJECT_DIR)

from Conformer_ASR.scripts.utils import config, Logger, save_manifest

TRAIN_PER = 0.9
TEST_PER = 0.1
LOGGER = Logger(name="PREPARING_MANIFESTS_COLLECTED")

class Dataset(Enum):
    Viettel = {"name": "Viettel",}
    Viettel_Assistant = {"name": "Viettel_Assistant",}
    Zalo = {"name": "Zalo"}
    FPT = {"name": "FPT"}

def create_train_test_manifests(dataset, config):
    if dataset == Dataset.Viettel:
        dataset_name = Dataset.Viettel["name"]
    if dataset == Dataset.Viettel_Assistant:
        dataset_name = Dataset.Viettel_Assistant["name"]
    elif dataset == Dataset.Zalo:
        dataset_name = Dataset.Zalo["name"]
    elif dataset == Dataset.FPT:
        dataset_name = Dataset.FPT["name"]
        
    dict_labels = {}
    train_dataset = []
    test_dataset = []
    
    LOGGER.log_info(f"Start prepare datasets for custom {dataset_name}")
    
    if os.path.exists(config.train_manifest): os.remove(config.train_manifest)
    if os.path.exists(config.test_manifest): os.remove(config.test_manifest)
    
    with open(config.manifest, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            data = json.loads(line)
            
            file_path = data["audio_filepath"]
            head, file_name = os.path.split(file_path)
            label = file_name.split("_")[0]
            
            if label in dict_labels.keys(): dict_labels[label].append(data)
            else: dict_labels[label] = [data]
            # manifest_data.append(data)
    
    for label, manifest_data in dict_labels.items():
        train_set, test_set = train_test_split(manifest_data, train_size=TRAIN_PER, test_size=TEST_PER, random_state=42)
        train_dataset.extend(train_set)
        test_dataset.extend(test_set)
    
    
    LOGGER.log_info(f"\tHas {len(train_dataset)} for training and {len(test_dataset)} for testing")
    save_manifest(config.train_manifest, train_dataset)
    save_manifest(config.test_manifest, test_dataset)
        
    LOGGER.log_info(f"Finish prepare datasets for custom {dataset_name}")
    print()
    
if __name__ == "__main__":
    config_viettel = config.get_config(["prepare_data", "viettel"])
    create_train_test_manifests(Dataset.Viettel, config_viettel)
    
    config_viettel_assistant = config.get_config(["prepare_data", "viettel_assistant"])
    create_train_test_manifests(Dataset.Viettel_Assistant, config_viettel_assistant)
    
    config_zalo = config.get_config(["prepare_data", "zalo"])
    create_train_test_manifests(Dataset.Zalo, config_zalo)
    
    config_fpt = config.get_config(["prepare_data", "fpt"])
    create_train_test_manifests(Dataset.FPT, config_fpt)