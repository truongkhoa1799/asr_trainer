import os
import torch
import librosa
import argparse
import numpy as np
import onnxruntime
import pytorch_lightning as ptl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from conformer_asr.utils import config, Logger, Config

LOGGER = Logger("CONVERT_CONFORMER_CTC")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def infer_signal(signal, ort_session, asr_model):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(asr_model.device)
    processed_signal, processed_signal_len = asr_model.preprocessor(
        input_signal=audio_signal, length=audio_signal_len,
    )
    print(f"processed_signal: {processed_signal.shape}")
    print(f"processed_signal_len: {processed_signal_len.shape}")
    # open('/home/khoa/Desktop/test.txt', 'w').write(str(list(processed_signal.detach().cpu().numpy()[0])))
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(processed_signal), 
        ort_session.get_inputs()[1].name: to_numpy(processed_signal_len), 
    }
    ologits = ort_session.run(None, ort_inputs)
    print(f"ologits: {ologits[0].shape}")
    alogits = np.asarray(ologits)
    logits = torch.from_numpy(alogits[0])
    greedy_predictions = logits.argmax(dim=-1, keepdim=False)
    
    current_hypotheses = asr_model._wer.ctc_decoder_predictions_tensor(
        greedy_predictions, 
        predictions_len=processed_signal_len, 
        return_hypotheses=False,
    )
    hypotheses = []
    hypotheses += current_hypotheses
    return hypotheses

# simple data layer to pass audio signal
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_model_path', default='', help='path for nemo model')
    parser.add_argument('--onnx_model_path', default='', help='path for onnx model')
    parser.add_argument('--device', default='cpu', help='device for loading model')
    args = parser.parse_args()
    
    if args.nemo_model_path == "" or args.onnx_model_path == "":
        LOGGER.log_info("invalid models path")
    
    LOGGER.log_info("Load Nemo model")
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.nemo_model_path,
        map_location=args.device
    )
    LOGGER.log_info("Done load Nemo model")
    
    LOGGER.log_info("Convert Nemo model")
    asr_model.export(
        args.onnx_model_path,
        onnx_opset_version=13
    )
    LOGGER.log_info("Done convert Nemo model")
    
    # LOGGER.log_info("Testing inference ONNX model")
    # ort_session = onnxruntime.InferenceSession(
    #     args.onnx_model_path,
    #     providers=['CPUExecutionProvider']
    # )
    
    # for input in ort_session.get_inputs(): print(f"input: {input}")
    # for output in ort_session.get_outputs(): print(f"output: {output}")
    # wav_path = "/home/khoatlv/ASR_Nemo/conformer_asr/convert/test_inference.wav"
    # data, sr = librosa.load(wav_path, sr=16000)
    # a = infer_signal(data, ort_session, asr_model)
    # print(a)
    