SOURCE_DIR="/home/khoatlv/ASR_Nemo"

LOG_NAME=$(echo $0 | cut -d'/' -f4 | cut -d'.' -f1)
LOG=$SOURCE_DIR/log/data_processing/"$LOG_NAME.txt"
if [ -e $LOG ];
then
    rm $LOG
fi

# CREATE TRAINING MANIFEST PAREMETERS
TRAINING_SET="VIVOS_TRAINING_SET"
TRAINING_DATA_DIRECTORY="/home/khoatlv/data/vivos/train"
TRAINING_MANIFEST_PATH="/home/khoatlv/data/vivos/manifests/vivos_train_manifest.json"
UNCLEANED_TRAINING_MANIFEST_PATH="/home/khoatlv/data/vivos/manifests/vivos_uncleaned_train_manifest.json"

# CREATE TESTING MANIFEST PAREMETERS
TESTING_SET="VIVOS_TESTING_SET"
TESTING_DATA_DIRECTORY="/home/khoatlv/data/vivos/test"
TESTING_MANIFEST_PATH="/home/khoatlv/data/vivos/manifests/vivos_test_manifest.json"
UNCLEANED_TESTING_MANIFEST_PATH="/home/khoatlv/data/vivos/manifests/vivos_uncleaned_test_manifest.json"

# EVALUATION DATASET PAREMETERS
LOG_DIR="/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data"

info() {
    command printf "%(%Y-%m-%d %T)T | INFO | %s\n" -1 "$@" 2>>1 | tee -a $LOG
}

info "-------------------------------------------------------------------"
info "---------------------- Prepare VIVOS Dataset ----------------------"
info "-------------------------------------------------------------------"
echo ""

info "----------- Create original training and testing manifest -----------"
python3 conformer_asr/data_processing/vivos/create_manifest_vivos.py \
    --dataset=$TRAINING_SET \
    --data_directory=$TRAINING_DATA_DIRECTORY \
    --manifest=$TRAINING_MANIFEST_PATH \
    --uncleaned_manifest=$UNCLEANED_TRAINING_MANIFEST_PATH | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot create original traing manifest"
    exit 1
fi
info ""

python3 conformer_asr/data_processing/vivos/create_manifest_vivos.py \
    --dataset=$TESTING_SET \
    --data_directory=$TESTING_DATA_DIRECTORY \
    --manifest=$TESTING_MANIFEST_PATH \
    --uncleaned_manifest=$UNCLEANED_TESTING_MANIFEST_PATH | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot create original testing manifest"
    exit 1
fi
info ""