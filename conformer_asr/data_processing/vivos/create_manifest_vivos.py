import os
import librosa
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import TypedDict
from functools import partial
from multiprocessing import Pool

from conformer_asr.utils import Logger, save_manifest, read_manifest, config, split_train_test_dataset, ManifestData

LOGGER = Logger(name="PREPARING_MANIFESTS_VIVOS")
vocabs = config.get_config(['training', 'vocabs'])

'''
python3 conformer_asr/data/create_manifest_vivos.py \
    --dataset="VIVOS_TRAINING_SET" \
    --data_directory="/home/khoatlv/data/vivos/train" \
    --manifest="/home/khoatlv/data/vivos/manifests/vivos_train_manifest.json" \
    --uncleaned_manifest="/home/khoatlv/data/vivos/manifests/vivos_uncleaned_train_manifest.json"
'''

class ManifestConfig(TypedDict):
    dataset: str
    data_directory: str
    manifest: str
    uncleaned_manifest: str

def check_oov(text):
    set_vocabs = set()
    for i in text:
        if i not in vocabs:
            return True, set_vocabs
        set_vocabs.add(i)
    return False, set_vocabs

def create_transcript_dict(transcript_path):
    transcript_dict = dict()
    with open(transcript_path, "r") as fin:
        for transcript_line in fin.readlines():
            transcript_line = transcript_line.replace("\n", "")
            transcript_line = transcript_line.split("|")
            
            audio_name = transcript_line[0].strip()
            transcript = transcript_line[1].lower().strip()
            
            transcript_dict[audio_name] = transcript
    return transcript_dict

def create_list_data_manifest(audio_path, transcript_dict):
    '''
    Create data manifest
    Return:
        - data: ManifestData
        - set_vocabs: set()
        - code: 
            - 0: success
            - 1: unmapped
            - 2: oov
            - 3: error
    '''
    try:
        audio_name, ext = os.path.splitext(audio_path)
        audio_name = audio_name.split("/")[-1]

        # Only allow audio in transcript_dict and in vocabs
        if audio_name not in transcript_dict:
            return str(audio_path), None, 1

        duration = librosa.get_duration(filename=str(audio_path))
        data = ManifestData(
            audio_filepath=str(audio_path),
            duration=round(duration, 2),
            text=transcript_dict[audio_name]
        )
        
        is_oov, set_vocabs = check_oov(transcript_dict[audio_name])
        if is_oov: 
            return data, set_vocabs, 2
        
        return data, set_vocabs, 0
    except Exception as e:
        print(e)
        return str(audio_path), None, 2
    
def create_manifest(config):
    '''
    Create training manifest for vivos dataset
        - Create transcript dictionary
        - Create manifest for training
            - If audio name is not in transcript_dict OR text of this audio has character out of vocabulary: => add unmapped list
            - Preprocessing text
            - Create set of vocabulary in this dataset
        - Summarize vocabulary in training dataset and original vocabulary
        - Save data manifest in path with config=prepare_data.vlsp2021.original_train_manifest
        - Save uncleaned data manifest in path with config=prepare_data.vlsp2021.uncleaned_manifest
    '''
    
    LOGGER.log_info(f"Create manifest for {config['dataset']}")
    manifest_path = Path(config['manifest'])
    uncleaned_manifest_path = Path(config['uncleaned_manifest'])
    
    # Create manifest directory if it doesn't exist
    if not manifest_path.parent.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
    data_manifests = []
    data_errors = []
    set_vocabularies = set()
    data_directory = Path(config['data_directory'])
        
    list_audio_path = list(data_directory.glob("wav/*.wav"))
    transcript_path = data_directory.joinpath("text")
        
    # Create transcript dict
    LOGGER.log_info(f"\tCreate transcript dict")
    transcript_dict = create_transcript_dict(transcript_path)
            
    LOGGER.log_info(f"\tNumber of audio files: {len(list_audio_path)}")
    LOGGER.log_info(f"\tNumber of transcript files: {len(transcript_dict.keys())}")
    LOGGER.log_info(f"\tDone create transcript dict")
    LOGGER.log_info("\tWrite audio and transcript to manifest")
        
    def iterator_audio_path(list_audio_path):
        for audio_path in list_audio_path:
            yield audio_path
                
    p = Pool(24)
    list_data_manifest = []
    list_audio_unmap_manifest = []
        
    iterator = iterator_audio_path(list_audio_path)
    partial_fn = partial(create_list_data_manifest, transcript_dict=transcript_dict)
    create_data_manifest_map = p.imap_unordered(
        partial_fn,
        tqdm(iterator, total=len(list_audio_path), desc="[Create data manifest]"),
        chunksize=10,
    )

    for data, set_vocabs, code in create_data_manifest_map:
        if code == 0: 
            set_vocabularies.update(set_vocabs)
            list_data_manifest.append(data)
        else:
            list_audio_unmap_manifest.append(data)
    
    # Map audio and transcript to create manifest
    data_manifests.extend(list_data_manifest)
    data_errors.extend(list_audio_unmap_manifest)
    
    print()
    del p
    
    diff_vocabularies = set_vocabularies - set(vocabs)
    save_manifest(manifest_path, data_manifests)
    save_manifest(uncleaned_manifest_path, data_errors)
    LOGGER.log_info(f"\tSave original manifest to {manifest_path}")
    LOGGER.log_info(f"\tSave uncleaned manifest to {uncleaned_manifest_path}")
    print()
    
    LOGGER.log_info(f"\tVocabularies differences: {diff_vocabularies}")
    LOGGER.log_info(f"\tNumber of audio processed: {len(data_manifests)}")
    LOGGER.log_info(f"\tNumber of audio rejected: {len(data_errors)}")
    LOGGER.log_info(f"Done create manifest for {config['dataset']}")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', help='name of evaluation dataset')
    parser.add_argument('--data_directory', default='', help='name of evaluation dataset')
    parser.add_argument('--manifest', default='', help='name of evaluation dataset')
    parser.add_argument('--uncleaned_manifest', default='', help='name of evaluation dataset')
    args = parser.parse_args()
    try:
        config = ManifestConfig(
            dataset=args.dataset,
            data_directory=args.data_directory,
            manifest=args.manifest,
            uncleaned_manifest=args.uncleaned_manifest,
        )
        create_manifest(config)
    except Exception as e:
        print(e)
        exit(-1)