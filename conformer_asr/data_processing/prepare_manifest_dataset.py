import os
import re
import sys
from collections import defaultdict
from conformer_asr.utils import config, Logger, read_manifest, save_manifest

LOGGER = Logger(name="CREATING_MANIFESTS_DATASETS")
vocabs = config.get_config(["training", "vocabs"])
vocabs_set = set(vocabs)
chars_to_ignore_regex   = '[\,\?\.\!\;\:\"\'\(\)\{\}\“\‘\”\…]'  # remove special character tokens

# ------------------------------------- Charaters ---------------------------------------------------------------
def get_charset(manifest_data):
    global vocabs
    charset = defaultdict(int)
    invalid_token = []
    for idx, row in enumerate(manifest_data):
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
        
        if idx % 10000 == 0: LOGGER.log_info(f"\t\tRead {idx} data rows")
    return charset, invalid_token

def extract_character_set(cfg, train_data_processed, test_data_processed):
    config = cfg.get_config(["training"])
    global train_set, test_set
    LOGGER.log_info("Start extract character set")

    train_charset, invalid_token_train = get_charset(train_data_processed)
    test_charset, invalid_token_test = get_charset(test_data_processed)
    LOGGER.log_info(f"\tInvalid token in train set{invalid_token_train}")
    LOGGER.log_info(f"\tInvalid token in test set{invalid_token_test}")

    train_set = set(train_charset.keys())
    test_set = set(test_charset.keys())

    LOGGER.log_info(f"\tNumber of tokens in vocabs set : {len(vocabs_set)}")
    LOGGER.log_info(f"\tVocabs Set{vocabs_set}")

    LOGGER.log_info(f"\tNumber of tokens in preprocessed train set : {len(train_set)}")
    LOGGER.log_info(f"\tNumber of tokens in preprocessed test set : {len(test_set)}")

    intersection_train_test = set.intersection(train_set, test_set)
    train_oov = train_set - intersection_train_test
    test_oov = test_set - intersection_train_test

    LOGGER.log_info(f"\tNumber of OOV tokens train set compared test set : {len(train_oov)} - {train_oov}")
    LOGGER.log_info(f"\tNumber of OOV tokens test set compared train set : {len(test_oov)} - {test_oov}")


    intersection_train_vocabs = set.intersection(train_set, vocabs_set)
    train_oov_vocabs = vocabs_set - intersection_train_vocabs
    LOGGER.log_info(f"\tNumber of OOV tokens vocabs set compared train set : {len(train_oov_vocabs)} - {train_oov_vocabs}")

    intersection_test_vocabs = set.intersection(test_set, vocabs_set)
    test_oov_vocabs = vocabs_set - intersection_test_vocabs
    LOGGER.log_info(f"\tNumber of OOV tokens vocabs set compared test set : {len(test_oov_vocabs)} - {test_oov_vocabs}")
    LOGGER.log_info("Finish extract character set")
    
# ------------------------------------- Preprocessing Data ---------------------------------------------------------------
def remove_special_characters(data):
    data["text"] = re.sub(chars_to_ignore_regex, '', data["text"]).lower().strip()
    return data

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in range(len(manifest)):
            manifest[idx] = processor(manifest[idx])
    return manifest

def preprocessing_data(config):
    manifest = config.get_config(["training", "manifest"])
    
    LOGGER.log_info("Start preprocessing data")
    PREPROCESSORS = [
        remove_special_characters,
    ]

    # Load manifests
    train_data, total_duration_train = read_manifest(manifest.train_manifest)
    test_data, total_duration_test = read_manifest(manifest.test_manifest)
    LOGGER.log_info(f"\tNumber of training data {len(train_data)} with total duration {round(total_duration_train/3600, 2)}")
    LOGGER.log_info(f"\tNumber of testing data {len(test_data)} with total duration {round(total_duration_test/3600, 2)}")

    # Apply preprocessing
    train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
    test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)

    # Write new manifests
    save_manifest(manifest.train_manifest_cleaned, train_data_processed)
    LOGGER.log_info(f"\tWrite preprocessing training data {manifest.train_manifest_cleaned}")
    save_manifest(manifest.test_manifest_cleaned, test_data_processed)
    LOGGER.log_info(f"\tWrite preprocessing testng data {manifest.test_manifest_cleaned}")
    
    LOGGER.log_info("Finish preprocessing data")
    print()
    
    return train_data_processed, test_data_processed

# ------------------------------------- Create manifests ---------------------------------------------------------------
def create_manifest(config):
    LOGGER.log_info("Starting creating manifest for training")
    
    test_script = "cat "
    train_script = "cat "
    
    prepare_data = config.get_config(["prepare_data"])
    config = config.get_config(["training"])
    
    datasets = prepare_data.keys()
    for dataset in datasets:
        if not config["use_datasets"][f"use_{dataset}"]: 
            continue
        LOGGER.log_info(f"\t{dataset.upper()}:")
        if dataset == "common_voice":
            train_manifest_path = prepare_data['common_voice']['train_manifest']
            dev_manifest_path = prepare_data['common_voice']['dev_manifest']
            test_manifest_path = prepare_data['common_voice']['test_manifest']
            
            num_training_data = os.popen(f"wc -l {train_manifest_path}").read().strip()
            num_dev_data = os.popen(f"wc -l {dev_manifest_path}").read().strip()
            num_testing_data = os.popen(f"wc -l {test_manifest_path}").read().strip()
            
            if not os.path.exists(train_manifest_path) \
                or not os.path.exists(dev_manifest_path) \
                or not os.path.exists(test_manifest_path):
                LOGGER.log_error(f"\tMissing manifests of {dataset} dataset")
                exit(-1)
            
            train_script += f"{train_manifest_path} {dev_manifest_path} "
            test_script += f"{test_manifest_path} "
            
            LOGGER.log_info(f"\t\tTrain:\t{num_training_data}")
            LOGGER.log_info(f"\t\tDev:\t{num_dev_data}")
            LOGGER.log_info(f"\t\tTest:\t{num_testing_data}")
            print()
        else:
            train_manifest_path = prepare_data[dataset].train_manifest
            test_manifest_path = prepare_data[dataset].test_manifest
            
            num_training_data = os.popen(f"wc -l {train_manifest_path}").read().strip()
            num_testing_data = os.popen(f"wc -l {test_manifest_path}").read().strip()
            
            if not os.path.exists(train_manifest_path) \
                or not os.path.exists(test_manifest_path):
                LOGGER.log_error(f"\tMissing manifests of {dataset} dataset")
                exit(-1)
            
            train_script += f"{train_manifest_path} "
            test_script += f"{test_manifest_path} "
            
            LOGGER.log_info(f"\t\tTrain:\t{num_training_data}")
            LOGGER.log_info(f"\t\tTest:\t{num_testing_data}")
            print()

    if os.path.exists(config.manifest.train_manifest): os.remove(config.manifest.train_manifest)
    if os.path.exists(config.manifest.test_manifest): os.remove(config.manifest.test_manifest)

    train_script += f"> {config.manifest.train_manifest}"
    test_script += f"> {config.manifest.test_manifest}"

    os.system(train_script)
    os.system(test_script)

    LOGGER.log_info("\tNumber of files in training and testing dataset")
    num_train_data = os.popen(f"wc -l {config.manifest.train_manifest}").read()
    num_test_data = os.popen(f"wc -l {config.manifest.test_manifest}").read()
    LOGGER.log_info(f"\t\t{num_train_data.strip()}")
    LOGGER.log_info(f"\t\t{num_test_data.strip()}")
    print()

# ------------------------------------- Main ---------------------------------------------------------------
# def remove_invalid_file():
#     manifest_data = []
#     with open("/home/khoatlv/ASR-NEMO/tokenizers/invalid_token_test.txt", 'r') as f:
#         for line in tqdm(f, desc="Reading manifest data"):
#             line = line.replace("\n", "")
#             data = json.loads(line)
#             manifest_data.append(data["data"])
#     with open("/home/khoatlv/ASR-NEMO/tokenizers/invalid_token_train.txt", 'r') as f:
#         for line in tqdm(f, desc="Reading manifest data"):
#             line = line.replace("\n", "")
#             data = json.loads(line)
#             manifest_data.append(data["data"])

#     count_has = 0
#     for i in manifest_data:
#         path = i["audio_filepath"] 
#         if os.path.exists(path): 
#             # os.remove(path)
#             count_has+=1

#     print(count_has)

'''
python3 conformer_asr/data/prepare_manifest_dataset.py
'''

if __name__ == "__main__":
    create_manifest(config)
    train_data_processed, test_data_processed = preprocessing_data(config)
    # extract_character_set(config, train_data_processed, test_data_processed)
    # remove_invalid_file()