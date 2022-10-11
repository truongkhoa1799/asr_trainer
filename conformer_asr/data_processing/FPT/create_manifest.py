import os
import librosa
import argparse
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import TypedDict

from conformer_asr.utils import Logger, save_manifest, read_manifest, config, split_train_test_dataset, ManifestData

LOGGER = Logger(name="PREPARING_MANIFESTS_FPT_Open_Datasets")
vocabs = config.get_config(['training', 'vocabs'])

class ManifestConfig(TypedDict):
    dataset: str
    wav_directory: str
    original_manifest: str
    uncleaned_manifest: str
    transcript_path: str

def create_transcript_dict(transcript_path):
    transcript_dict = dict()
    with open(transcript_path, "r") as fin:
        for transcript_line in fin.readlines():
            transcript_line = transcript_line.replace("\n", "")
            transcript_line = transcript_line.split("|")
            
            audio_name = transcript_line[0].strip()
            audio_name, ext = os.path.splitext(audio_name)
            
            transcript = transcript_line[1].lower().strip()
            
            transcript_dict[audio_name] = transcript
    return transcript_dict

def check_oov(text):
    set_vocabs = set()
    for i in text:
        if i not in vocabs:
            return True, set_vocabs
        set_vocabs.add(i)
    return False, set_vocabs

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
    Create training manifest for vlsp_2021 dataset
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
    manifest_path = Path(config['original_manifest'])
    uncleaned_manifest_path = Path(config['uncleaned_manifest'])
    wav_directory = Path(config['wav_directory'])
    transcript_path = Path(config['transcript_path'])
    
    # Create manifest directory if it doesn't exist
    if not manifest_path.parent.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
    data_manifests = []
    data_errors = []
    set_vocabularies = set()    
    audio_paths = list(wav_directory.glob("*.wav"))
    
    # Create transcript dict
    LOGGER.log_info(f"\tCreate transcript dict")
    transcript_dict = create_transcript_dict(transcript_path)
    print(list(transcript_dict.keys())[0])
    
    LOGGER.log_info(f"\tNumber of audio files: {len(audio_paths)}")
    LOGGER.log_info(f"\tNumber of transcript files: {len(transcript_dict.keys())}")
    LOGGER.log_info(f"\tDone create transcript dict")
    LOGGER.log_info("\tWrite audio and transcript to manifest")
    
    def iterator_audio_path(audio_paths):
        for audio_path in audio_paths:
            yield audio_path
                
    p = Pool(24)
    iterator = iterator_audio_path(audio_paths)
    partial_fn = partial(create_list_data_manifest, transcript_dict=transcript_dict)
    create_data_manifest_map = p.imap_unordered(
        partial_fn,
        tqdm(iterator, total=len(audio_paths), desc="[Create data manifest]"),
        chunksize=10,
    )

    for data, set_vocabs, code in create_data_manifest_map:
        if code == 0: 
            set_vocabularies.update(set_vocabs)
            data_manifests.append(data)
        else:
            data_errors.append(data)
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
    parser.add_argument('--wav_directory', default='', help='name of evaluation dataset')
    parser.add_argument('--original_manifest', default='', help='name of evaluation dataset')
    parser.add_argument('--uncleaned_manifest', default='', help='name of evaluation dataset')
    parser.add_argument('--transcript_path', default='', help='name of evaluation dataset')
    args = parser.parse_args()
    try:
        config = ManifestConfig(
            dataset=args.dataset,
            wav_directory=args.wav_directory,
            original_manifest=args.original_manifest,
            uncleaned_manifest=args.uncleaned_manifest,
            transcript_path=args.transcript_path
        )
        create_manifest(config)
    except Exception as e:
        print(e)
        exit(-1)
    
