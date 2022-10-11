SOURCE_DIR="/home/khoatlv/ASR_Nemo"

LOG_NAME=$(echo $0 | cut -d'/' -f4 | cut -d'.' -f1)
LOG=$SOURCE_DIR/log/data_processing/"$LOG_NAME.txt"
if [ -e $LOG ];
then
    rm $LOG
fi

info() {
    command printf "%(%Y-%m-%d %T)T | INFO | %s\n" -1 "$@" 2>>1 | tee -a $LOG
}


info "------------------------------------------------------------------"
info "-------------------- PROCESS FPT OPEN DATASET --------------------"
info "------------------------------------------------------------------"
info ""

BASE_DIR="/home/khoatlv/ASR_Nemo"
DATASET="FPT_Open_Datasets"
ALL_TEXTS=$PROCESSED_DATA_DIR"/all_transcripts.txt"
PROCESSED_DATA_DIR="/home/khoatlv/data/FPT/processed_data"
MANIFEST_PATH="/home/khoatlv/data/FPT/manifests/manifests.json"
ORIGINAL_MANIFEST_PATH="/home/khoatlv/data/FPT/manifests/original_manifests.json"
UNCLEANED_MANIFEST_PATH="/home/khoatlv/data/FPT/manifests/uncleaned_manifests.json"

TRAIN_MANIFEST_PATH="/home/khoatlv/data/FPT/manifests/fpt_train_manifest.json"
TEST_MANIFEST_PATH="/home/khoatlv/data/FPT/manifests/fpt_test_manifest.json"

# Default datasets
MP3_DIR="/home/khoatlv/data/FPT/mp3"
WAV_DIR="/home/khoatlv/data/FPT/wav"
TRANSCRIPT_PATH="/home/khoatlv/data/FPT/transcriptAll.txt"

# EVALUATION DATASET PAREMETERS
LOG_DIR="/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data"

# if [ -e $ALL_TEXTS ]; then
#     rm $ALL_TEXTS
# fi

# if [ -e $PROCESSED_DATA_DIR ]; then
#     rm -rf $PROCESSED_DATA_DIR
#     mkdir -p $PROCESSED_DATA_DIR
# else
#     mkdir -p $PROCESSED_DATA_DIR
# fi

# info "----------------- Separate clean and noise text ------------------"
# python3 conformer_asr/data_processing/FPT/separate_clean_dataset.py TRANSCRIPT_PATH=$TRANSCRIPT_PATH PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot separate_clean_dataset"
#     info 1
# else
#     info "Clean dataset "$TRANSCRIPT_PATH
# fi
# info ""

# info "----------------- Process dataset has digit ------------------"
# python3 conformer_asr/data_processing/FPT/clean_digit_dataset.py PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot process dataset has digit"
#     exit 1
# else
#     info "Done process dataset has digit "$PROCESSED_DATA_DIR
# fi
# info ""

# info "----------------- Process dataset has OOV ------------------"
# python3 conformer_asr/data_processing/FPT/clean_oov_dataset.py PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot process dataset has OOV"
#     exit 1
# else
#     info "Done process dataset has OOV "$PROCESSED_DATA_DIR | tee -a $LOG
# fi
# info ""

# info "----------------- Concatenate all clean data ------------------"
# for data in $PROCESSED_DATA_DIR/clean/*; do
#     cat $data >> $ALL_TEXTS
# done

# info "----------------- Convert MP3 to Wav format ------------------"
# python3 conformer_asr/data_processing/FPT/convert_audios.py MP3_DIR=$MP3_DIR WAV_DIR=$WAV_DIR | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot Convert MP3 to Wav format"
#     exit 1
# else
#     info "Done Convert MP3 to Wav format in "$WAV_DIR
# fi
# info ""

# info "----------- Create original manifest -----------"
# python3 conformer_asr/data_processing/FPT/create_manifest.py \
#     --dataset=$DATASET \
#     --wav_directory=$WAV_DIR \
#     --original_manifest=$ORIGINAL_MANIFEST \
#     --uncleaned_manifest=$UNCLEANED_MANIFEST_PATH \
#     --transcript_path=$ALL_TEXTS | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot create original manifest"
#     exit 1
# fi
# info ""

# info "------------------- Evaluating manifest -------------------"
# python3 conformer_asr/evaluation/evaluate_asr_data.py -e \
#     --original_manifest_path=$ORIGINAL_MANIFEST_PATH \
#     --dataset_name=$DATASET \
#     --log_dir=$LOG_DIR | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot Evaluating training manifest"
#     exit 1
# fi
# info ""


# info "-------------------- Cleaning manifest ---------------------"
# python3 conformer_asr/evaluation/evaluate_asr_data.py -c \
#     --manifest_path=$MANIFEST_PATH \
#     --original_manifest_path=$ORIGINAL_MANIFEST_PATH \
#     --dataset_name=$DATASET \
#     --log_dir=$LOG_DIR | tee -a $LOG
# if [ $? -eq 255 ];
# then
#     info "Cannot cleaning training manifest"
#     exit 1
# fi
# info ""

info "-------------------- Split manifest into training and testing ---------------------"
python3 conformer_asr/data_processing/FPT/split_dataset.py \
    --manifest_path=$MANIFEST_PATH \
    --train_manifest_path=$TRAIN_MANIFEST_PATH \
    --test_manifest_path=$TEST_MANIFEST_PATH | tee -a $LOG
if [ $? -eq 255 ];
then
    info "Cannot split manifest into training and testing"
    exit 1
fi
info ""