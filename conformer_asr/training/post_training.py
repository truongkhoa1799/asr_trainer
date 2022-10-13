import torch
from pathlib import Path
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

'''
POST TRAINING:
    - CONVERT CHECKPOINT TO NEMO MODEL
    - CONVERT NEMO TO ONNX MODEL
'''

# CHECKPOINT
experiments_path=Path("/home/khoatlv/ASR_Nemo/experiments/old_checkpoint")
model_path="2022-10-11_tokenizer_512"
checkpoint_name="Conformer_small_Model_Language_vi--val_wer=0.0826-epoch=3-last.ckpt"
checkpoint_path=experiments_path.joinpath(model_path, checkpoint_name)

# NEMO 
nemo_model_name=checkpoint_name.replace('.ckpt', '.nemo')
nemo_model_path=experiments_path.joinpath(model_path, nemo_model_name)

# Pytorch model 
torch_model_name=checkpoint_name.replace('.ckpt', '.pt')
torch_model_path=experiments_path.joinpath(model_path, nemo_model_name)

def convert_checkpoint(checkpoint_path, nemo_model_path, torch_model_path):
    model = EncDecCTCModel.load_from_checkpoint(checkpoint_path, map_locations='cpu', strict=False)
    
    # We can update model config in this scope
    # model.cfg.tokenizer.dir='/home/khoa/NovaIntechs/src/Smart-Speaker-Common/smart_speaker_common/modules/conformer/Conformer_tokenizer_512_v2'
    # model.cfg.tokenizer.model_path='/home/khoa/NovaIntechs/src/Smart-Speaker-Common/smart_speaker_common/modules/conformer/Conformer_tokenizer_512_v2/tokenizer_spe_bpe_v512/tokenizer.model'
    # model.cfg.tokenizer.vocab_path='/home/khoa/NovaIntechs/src/Smart-Speaker-Common/smart_speaker_common/modules/conformer/Conformer_tokenizer_512_v2/tokenizer_spe_bpe_v512/vocab.txt'
    # model.cfg.tokenizer.spe_tokenizer_vocab='/home/khoa/NovaIntechs/src/Smart-Speaker-Common/smart_speaker_common/modules/conformer/Conformer_tokenizer_512_v2/tokenizer_spe_bpe_v512/tokenizer.vocab'

    # Save model to nemo model path
    model.save_to(nemo_model_path)
    
    # # Convert nemo model to pt
    # torch_model = EncDecCTCModelBPE.restore_from(
    #     restore_path=nemo_model_path,
    #     map_location='cpu'
    # )
    # torch_model.export(torch_model_path)
    
if __name__ == '__main__':
    convert_checkpoint(checkpoint_path, nemo_model_path, torch_model_path)