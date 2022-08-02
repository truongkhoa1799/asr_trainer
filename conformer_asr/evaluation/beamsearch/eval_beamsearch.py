import nemo.collections.asr as nemo_asr
import torch
import numpy as np
import pytorch_lightning as ptl
from omegaconf import OmegaConf
import os
import librosa
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader

def load_config(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    return config

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

data_layer = AudioDataLayer(sample_rate=16000)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

# Conformer
# model_path = "/home/nhan/NovaIntechs/models/conformer_small/Conformer_small_Model_Language_vi--val_wer=0.0426-epoch=98.ckpt"
# config_path = "/home/khoatlv/ASR-NEMO/config/conformer_small_ctc_bpe.yaml"
# config = load_config(config_path)
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from("/home/khoatlv/ASR-NEMO/models/conformer/Conformer_small_epoch=98.nemo")
asr_model.eval()
lm_path = "/home/khoatlv/ASR-NEMO/n_gram_lm/n_gram_lm_model/6-conformer-small-gram_trained.bin"

asr_model.beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
    vocab=list(asr_model.decoder.vocabulary),
    beam_width=200,
    alpha=2, beta=2.5,
    lm_path=lm_path,
    num_cpus=max(os.cpu_count(), 1),
    input_tensor=False
)

audio_name = "FPTOpenSpeechData_Set002_V0.1_011692.wav"
AUDIO_FILENAME = os.path.join("/home/khoatlv/data/FPT/wav", audio_name)
signal, sr = librosa.load(AUDIO_FILENAME, sr=16000)

# softmax implementation in NumPy
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

# let's do inference once again but without decoder
logits = asr_model.transcribe([AUDIO_FILENAME], logprobs=True)[0]
probs = softmax(logits)
preds = np.argmax(probs, axis=1)
# a = asr_model.beam_search_lm.forward(log_probs = np.expand_dims(probs, axis=0), log_probs_length=None)
# print(a)
print(preds)



# data_layer.set_signal(signal)
# batch = next(iter(data_loader))
# audio_signal, audio_signal_len = batch
# hypotheses = asr_model.infer_signal(audio_signal, audio_signal_len)
# # print(hypotheses)

# # Convert our audio sample to text
# files = [AUDIO_FILENAME]
# transcript = asr_model.transcribe(paths2audio_files=files)[0]
# print(f'Transcript: "{transcript}"')

