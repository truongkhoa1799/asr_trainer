import os
import torch
import librosa
import argparse
import numpy as np
import nemo.collections.asr as nemo_asr

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def infer_signal(signal, model, asr_model):
    audio_signal, audio_signal_len = torch.as_tensor(np.array([signal]), dtype=torch.float32), torch.as_tensor(np.array([signal.size]), dtype=torch.int64)
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(asr_model.device)
    
    processed_signal, processed_signal_len = asr_model.preprocessor(
        input_signal=audio_signal, length=audio_signal_len,
    )
    # print(f"processed_signal: {processed_signal.shape}")
    # print(f"processed_signal_len: {processed_signal_len.shape}")
    
    np.savez("preprocess_input.npz", processed_signal=processed_signal, processed_signal_len=processed_signal_len)
    encoded, encoded_len = model.encoder(audio_signal=processed_signal, length=processed_signal_len)
    print(f"encoded: {encoded.shape}")
    
    # alogits = np.asarray(ologits)
    # logits = torch.from_numpy(alogits[0])
    # greedy_predictions = logits.argmax(dim=-1, keepdim=False)
    # current_hypotheses = asr_model._wer.decoding.ctc_decoder_predictions_tensor(
    #     greedy_predictions, 
    #     decoder_lengths=processed_signal_len, 
    #     return_hypotheses=False,
    # )
    # hypotheses = []
    # hypotheses += current_hypotheses
    # return hypotheses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_model_path', required=False, default='models/conformer/Conformer_small_Model_Language_vi_epoch=250.nemo', help='path for nemo model')
    parser.add_argument('--onnx_model_path', required=False, default='', help='path for onnx model')
    parser.add_argument('--device', default='cpu', help='device for loading model')
    args = parser.parse_args()
    
    print("Load Nemo model")
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.nemo_model_path,
        map_location=args.device
    )
    asr_model.eval()
    print("Done load Nemo model")
    
    wav_path = "/home/khoatlv/ASR_Nemo/conformer_asr/convert/test_inference.wav"
    data, sr = librosa.load(wav_path, sr=16000)
    a = infer_signal(data, asr_model, asr_model)
    