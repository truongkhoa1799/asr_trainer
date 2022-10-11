SOURCE_DIR="/home/khoatlv/ASR_Nemo"

LOG_NAME=$(echo $0 | cut -d'/' -f4 | cut -d'.' -f1)
LOG=$SOURCE_DIR/log/data_processing/"$LOG_NAME.txt"
if [ -e $LOG ];
then
    rm $LOG
fi

# CREATE TRAINING MANIFEST PAREMETERS
TRAINING_SET="VLSP_2021_TRAINING_SET"
TRAINING_DATA_DIRECTORY="/home/khoatlv/data/vlsp2021/training_set"
TRAINING_MANIFEST_PATH="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_train_manifest.json"
ORIGINAL_TRAIN_MANIFEST_PATH="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_original_train_manifest.json"
UNCLEANED_TRAIN_MANIFEST_PATH="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_uncleaned_train_manifest.json"

# CREATE TESTING MANIFEST PAREMETERS
TESTING_SET="VLSP_2021_TESTING_SET"
TESTING_DATA_DIRECTORY="/home/khoatlv/data/vlsp2021/private_test"
TESTING_MANIFEST_PATH="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_test_manifest.json"
ORIGINAL_TEST_MANIFEST_PATH="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_original_test_manifest.json"
UNCLEANED_TEST_MANIFEST_PATH="/home/khoatlv/data/vlsp2021/manifests/vlsp2021_uncleaned_test_manifest.json"

# EVALUATION DATASET PAREMETERS
LOG_DIR="/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data"

info() {
    command printf "%(%Y-%m-%d %T)T | INFO | %s\n" -1 "$@" 2>>1 | tee -a $LOG
}

info "-------------------------------------------------------------------"
info "-------------------- Prepare VLSP 2021 Dataset --------------------"
info "-------------------------------------------------------------------"
echo ""
info "1. From list audios and file contianing all transcripts of training and testing data"
info "  - Create manifest for training and testing"
info "  - Remove all file which has OOV"
info "2. Evaluating training and testing data using Wav2Vec model"
info "  - Create error file and evaluation file"
info "3. Create final training and testing manifest by remove all data with WER > threshold and in error result"

echo ""

info "----------- Create original training and testing manifest -----------"
python3 conformer_asr/data_processing/vlsp2021/create_manifest_vlsp2021.py \
    --dataset=$TRAINING_SET \
    --data_directory=$TRAINING_DATA_DIRECTORY \
    --original_manifest=$ORIGINAL_TRAIN_MANIFEST_PATH \
    --uncleaned_manifest=$UNCLEANED_TRAIN_MANIFEST_PATH | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot create original traing manifest"
    exit 1
fi
info ""

python3 conformer_asr/data_processing/vlsp2021/create_manifest_vlsp2021.py \
    --dataset=$TESTING_SET \
    --data_directory=$TESTING_DATA_DIRECTORY \
    --original_manifest=$ORIGINAL_TEST_MANIFEST_PATH \
    --uncleaned_manifest=$UNCLEANED_TEST_MANIFEST_PATH | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot create original testing manifest"
    exit 1
fi
info ""


info "------------------- Evaluating training and testing manifest -------------------"
python3 conformer_asr/evaluation/evaluate_asr_data.py -e \
    --original_manifest_path=$ORIGINAL_TRAIN_MANIFEST_PATH \
    --dataset_name=$TRAINING_SET \
    --log_dir=$LOG_DIR | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot Evaluating training manifest"
    exit 1
fi
info ""

python3 conformer_asr/evaluation/evaluate_asr_data.py -e \
    --original_manifest_path=$ORIGINAL_TEST_MANIFEST_PATH \
    --dataset_name=$TESTING_SET \
    --log_dir=$LOG_DIR | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot Evaluating testing manifest"
    exit 1
fi
info ""


info "-------------------- Cleaning training and testing manifest ---------------------"
python3 conformer_asr/evaluation/evaluate_asr_data.py -c \
    --manifest_path=$TRAINING_MANIFEST_PATH \
    --original_manifest_path=$ORIGINAL_TRAIN_MANIFEST_PATH \
    --dataset_name=$TRAINING_SET \
    --log_dir=$LOG_DIR | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot cleaning training manifest"
    exit 1
fi
info ""

python3 conformer_asr/evaluation/evaluate_asr_data.py -c \
    --manifest_path=$TESTING_MANIFEST_PATH \
    --original_manifest_path=$ORIGINAL_TEST_MANIFEST_PATH \
    --dataset_name=$TESTING_SET \
    --log_dir=$LOG_DIR | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot cleaning testing manifest"
    exit 1
fi
info ""