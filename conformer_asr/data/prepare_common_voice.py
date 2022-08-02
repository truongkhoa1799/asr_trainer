import csv
import os
import json
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
import librosa

data_dir = '/home/nhan/NovaIntechs/data/ASR_Data/common_voice'
manifest_dir = '/home/nhan/NovaIntechs/data/ASR_Data/common_voice/manifests'

wav_path = os.path.join(data_dir, "clips")
new_wave_path = os.path.join(data_dir, "wav")

train_manifest = f"{manifest_dir}/commonvoice_train_manifest.json"
dev_manifest = f"{manifest_dir}/commonvoice_dev_manifest.json"
test_manifest = f"{manifest_dir}/commonvoice_test_manifest.json"

def convert_mp3_to_wav():
    count = 0
    for mp3_file in os.listdir(wav_path):
        file_name = mp3_file.split('.')
        f = os.path.join(wav_path, mp3_file)
        if os.path.isfile(f):
            new_path = os.path.join(new_wave_path, file_name[0] + '.wav')
            sound = AudioSegment.from_mp3(f)
            sound = sound.set_frame_rate(16000)
            sound.export(new_path, format="wav")
            count += 1

        if count % 100 == 0: print("Processed {} files".format(count))

def create_manifest():
    for csv_file in ['train.tsv', 'dev.tsv', 'test.tsv']:
        csv_file_path = os.path.join(data_dir, csv_file)
        output_name=f'commonvoice_{os.path.splitext(csv_file)[0]}_manifest.json'
        output_file = Path(manifest_dir) / output_name
        output_file.parent.mkdir(exist_ok=True, parents=True)
        count_invalid = 0

        text_dict = dict()
        with open(csv_file_path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                text_dict[row['path'].strip()] = row['sentence'].strip()


        with output_file.open(mode='w', encoding='utf-8') as f:
            for i in os.listdir(new_wave_path):
                file_path = os.path.join(new_wave_path, i)
                filename, ext = os.path.splitext(i)
                new_filename = filename + ".mp3"
                if not new_filename in text_dict.keys(): continue
                data, sample_rate = sf.read(file_path)
                if sample_rate != 16000: count_invalid+=1
                duration = librosa.get_duration(y=data, sr=sr)

                f.write(
                    json.dumps({'audio_filepath': file_path, "duration": duration, 'text': text_dict[new_filename].strip()}, ensure_ascii=False) + '\n'
                )


# convert_mp3_to_wav()
# create_manifest()