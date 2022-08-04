import os
import librosa
from pathlib import Path
from conformer_asr.utils import Logger, save_manifest, read_manifest, config, split_train_test_dataset, ManifestData

LOGGER = Logger(name="PREPARING_MANIFESTS_VLSP_2021")

def create_transcript_dict(transcript_path):
    transcript_dict = dict()
    with open(transcript_path, "r") as fin:
        for transcript_line in fin.readlines():
            transcript_line = transcript_line.replace("\n", "")
            transcript_line = transcript_line.split("\t")
            
            audio_name = transcript_line[0].strip()
            transcript = transcript_line[1].strip()
            
            transcript_dict[audio_name] = transcript
            
            # break
    
    return transcript_dict
    
def create_list_data_manifest(list_audio_path, transcript_dict):
    list_audio_unmap = []
    list_data_manifest = []
    for idx, audio_path in enumerate(list_audio_path):
        audio_name, ext = os.path.splitext(audio_path)
        audio_name = audio_name.split("/")[-1]
    
        if audio_name not in transcript_dict:
            list_audio_unmap.append(str(audio_path))
            continue
    
        duration = librosa.get_duration(filename=str(audio_path))
        data = ManifestData(
            audio_filepath=str(audio_path),
            duration=duration,
            text=transcript_dict[audio_name]
        )
        list_data_manifest.append(data)
    
    return list_data_manifest, list_audio_unmap

def create_training_manifest(config):
    LOGGER.log_info("Create training manifest for VLSP 2021")
    train_manifest_path = Path(config.train_manifest)
    
    # Create manifest directory if it doesn't exist
    if not train_manifest_path.parent.exists():
        train_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
    training_data = []
    list_audio_unmap = []
    train_directory = Path(config.train_directory)
    for train_set in train_directory.iterdir():
        LOGGER.log_info(f"\tCreating traing manifest for train set {train_set.name}")
        
        list_audio_path = list(train_set.glob("wav/*.wav"))
        transcript_path = train_set.joinpath("transcript.txt")
        
        # Create transcript dict
        LOGGER.log_info(f"\t\tCreate transcript dict")
        transcript_dict = create_transcript_dict(transcript_path)
            
        LOGGER.log_info(f"\t\tNumber of audio files: {len(list_audio_path)}")
        LOGGER.log_info(f"\t\tNumber of transcript files: {len(transcript_dict.keys())}")
        LOGGER.log_info(f"\t\tDone create transcript dict")
        
        # Map audio and transcript to create manifest
        LOGGER.log_info("\t\tWrite audio and transcript to manifest")
        tmp_data_manifest, tmp_list_audio_unmap = create_list_data_manifest(list_audio_path, transcript_dict)
        training_data.extend(tmp_data_manifest)
        list_audio_unmap.extend(tmp_list_audio_unmap)
        
        LOGGER.log_info(f"\tDone create training manifest for train set {train_set.name}")
    
    save_manifest(train_manifest_path, training_data)
    LOGGER.log_info(f"\tNumber of training audio: {len(training_data)}")
    LOGGER.log_info(f"\tNumber of unmapped audio and transcipt: {len(list_audio_unmap)}")
    LOGGER.log_info(f"\tUnmapped audio and transcipt: {list_audio_unmap}")
    LOGGER.log_info("Done create testing manifest for VLSP 2021")
    
def create_testing_manifest(config):
    LOGGER.log_info("Create testing manifest for VLSP 2021")
    test_manifest_path = Path(config.test_manifest)
    
    # Create manifest directory if it doesn't exist
    if not test_manifest_path.parent.exists():
        test_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # create manifest for private test
    testing_data = []
    list_audio_unmap = []
    test_directory = Path(config.test_directory)
    for test_set in test_directory.iterdir():
        LOGGER.log_info(f"\tCreating test manifest for private test {test_set.name}")
        
        list_audio_path = list(test_set.glob("wav/*.wav"))
        transcript_path = test_set.joinpath("transcript.txt")
        
        # Create transcript dict
        LOGGER.log_info(f"\t\tCreate transcript dict")
        transcript_dict = create_transcript_dict(transcript_path)
            
        LOGGER.log_info(f"\t\tNumber of audio files: {len(list_audio_path)}")
        LOGGER.log_info(f"\t\tNumber of transcript files: {len(transcript_dict.keys())}")
        LOGGER.log_info(f"\t\tDone create transcript dict")
        
        # Map audio and transcript to create manifest
        LOGGER.log_info("\t\tWrite audio and transcript to manifest")
        tmp_data_manifest, tmp_list_audio_unmap = create_list_data_manifest(list_audio_path, transcript_dict)
        testing_data.extend(tmp_data_manifest)
        list_audio_unmap.extend(tmp_list_audio_unmap)
        LOGGER.log_info("\t\tDone write audio and transcript to manifest")
        
        LOGGER.log_info(f"\tDone create testing manifest for private test {test_set.name}")
            
    save_manifest(test_manifest_path, testing_data)
    LOGGER.log_info(f"\tNumber of testing audio: {len(testing_data)}")
    LOGGER.log_info(f"\tNumber of unmapped audio and transcipt: {len(list_audio_unmap)}")
    LOGGER.log_info(f"\tUnmapped audio and transcipt: {list_audio_unmap}")
    LOGGER.log_info("Done create testing manifest for VLSP 2021")
        
if __name__ == '__main__':
    vlsp_2021_config = config.get_config(['prepare_data', 'vlsp2021'])
    create_training_manifest(vlsp_2021_config)
    create_testing_manifest(vlsp_2021_config)
    