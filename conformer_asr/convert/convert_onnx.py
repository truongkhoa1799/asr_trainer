import os
import torch
import librosa
import argparse
import numpy as np
# import onnxruntime
import pytorch_lightning as ptl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from conformer_asr.utils import config, Logger, Config

'''
python3 /home/khoatlv/ASR_Nemo/conformer_asr/convert/convert_onnx.py \
    --nemo_model_path="/home/khoatlv/ASR_Nemo/experiments/old_checkpoint/2022-09-01/epoch_200.nemo" \
    --onnx_model_path="/home/khoatlv/ASR_Nemo/models/conformer/Conformer_epoch_200_v2.onnx" \
    --device='cuda'
'''

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def infer_signal(signal, ort_session, asr_model):
    audio_signal, audio_signal_len = torch.as_tensor(np.array([signal]), dtype=torch.float32), torch.as_tensor(np.array([signal.size]), dtype=torch.int64)
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(asr_model.device)
    
    processed_signal, processed_signal_len = asr_model.preprocessor(
        input_signal=audio_signal, length=audio_signal_len,
    )
    # print(f"processed_signal: {processed_signal.shape}")
    # print(f"processed_signal_len: {processed_signal_len.shape}")
    
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(processed_signal), 
        ort_session.get_inputs()[1].name: to_numpy(processed_signal_len), 
    }
    ologits = ort_session.run(None, ort_inputs)
    # print(f"ologits: {ologits[0].shape}")
    
    alogits = np.asarray(ologits)
    logits = torch.from_numpy(alogits[0])
    greedy_predictions = logits.argmax(dim=-1, keepdim=False)
    current_hypotheses = asr_model._wer.decoding.ctc_decoder_predictions_tensor(
        greedy_predictions, 
        decoder_lengths=processed_signal_len, 
        return_hypotheses=False,
    )
    hypotheses = []
    hypotheses += current_hypotheses
    return hypotheses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_model_path', default='', help='path for nemo model')
    parser.add_argument('--onnx_model_path', default='', help='path for onnx model')
    parser.add_argument('--device', default='cpu', help='device for loading model')
    args = parser.parse_args()
    
    if args.nemo_model_path == "" or args.onnx_model_path == "":
        print("invalid models path")
    
    print("Load Nemo model")
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.nemo_model_path,
        map_location=args.device
    )
    # print("Done load Nemo model")
    # print(asr_model.state_dict().keys())
    # torch.save(asr_model.state_dict(), "/home/khoatlv/ASR_Nemo/models/conformer_250")
    
    print("Convert Nemo model")
    asr_model.export(
        args.onnx_model_path,
        onnx_opset_version=13
    )
    print("Done convert Nemo model")
    
    # ort_session = onnxruntime.InferenceSession(
    #     args.onnx_model_path,
    #     providers=['CPUExecutionProvider']
    # )
    
    # print("Input ONNX model:")
    # for input in ort_session.get_inputs(): 
    #     print(f"\t{input}")
        
    # print("Output ONNX model:")
    # for output in ort_session.get_outputs():
    #     print(f"\t{input}")
        
    # print("Testing inference ONNX model")
    # wav_path = "/home/khoatlv/ASR_Nemo/conformer_asr/convert/test_inference.wav"
    # data, sr = librosa.load(wav_path, sr=16000)
    # a = infer_signal(data, ort_session, asr_model)
    # print(f"Testing inference ONNX model result: {a[0][0]}")
    