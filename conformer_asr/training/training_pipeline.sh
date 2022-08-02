#!/bin/sh

SOURCE_DIR="/home/khoatlv/Conformer_ASR"
PREPARE_DATASETS_DIR=$SOURCE_DIR/scripts/data

TRAINING_LOG=$SOURCE_DIR/log/trainig.txt
info() {
    command printf "%(%Y-%m-%d %T)T | INFO | %s\n" -1 "$@" 2>>1 | tee -a $TRAINING_LOG
}

python3() {
    command python3 "$@" 2>>1 | tee -a $TRAINING_LOG
}

rm $TRAINING_LOG
declare -a prepare_data_python=(
    "prepare_infore.py"
    "prepare_manifest_data_collected.py"
    "prepare_vlsp2020.py"
)

# # START PREPARE DATASETS
# info "START PREPARE DATASETS"
# for i in "${prepare_data_python[@]}"
# do
#     python_script=$PREPARE_DATASETS_DIR/$i
#     python3 $python_script
#     if (( $? == -1 ))
#     then
#         info "Fail to prepare data in $python_script"
#     fi
# done

# START CREATING MANIFEST DATASETS
info "START CREATING MANIFEST DATASETS"
python_script=$PREPARE_DATASETS_DIR/"create_manifest_dataset.py"
python3 $python_script
if (( $? == -1 ))
then
    info "Fail to prepare data in $python_script"
fi

# START TRAINGING CONFORMER
info "START TRAINGING CONFORMER"
python_script=$SOURCE_DIR/"train_conformer.py"
python3 $python_script
if (( $? == -1 ))
then
    info "Fail to prepare data in $python_script"
fi