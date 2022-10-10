echo "------------------------------------------------------------------"
echo "-------------------- PROCESS FPT OPEN DATASET --------------------"
echo "------------------------------------------------------------------"
echo ""

BASE_DIR="/home/khoatlv/ASR_Nemo"
PROCESSED_DATA_DIR=$BASE_DIR"/conformer_asr/data_processing/FPT"
ALL_TEXTS=$BASE_DIR"/data/all_transcripts.txt"

MP3_DIR="/home/khoa/NovaIntechs/data/ASR/FPT_Open_Speech_Dataset/mp3"
WAV_DIR="/home/khoa/NovaIntechs/data/ASR/FPT_Open_Speech_Dataset/wav"
TRANSCRIPT_PATH="/home/khoa/NovaIntechs/data/ASR/FPT_Open_Speech_Dataset/transcriptAll.txt"

# if [ -e $ALL_TEXTS ]; then
#     rm $ALL_TEXTS
# fi

# if [ -e $PROCESSED_DATA_DIR ]; then
#     rm -rf $PROCESSED_DATA_DIR
#     mkdir -p $PROCESSED_DATA_DIR
# else
#     mkdir -p $PROCESSED_DATA_DIR
# fi

# echo "----------------- Separate clean and noise text ------------------"
# python3 separate_clean_dataset.py TRANSCRIPT_PATH=$TRANSCRIPT_PATH PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR
# if [ $? -eq 255 ];
# then
#     echo "Cannot separate_clean_dataset"
#     exit 1
# else
#     echo "Clean dataset "$TRANSCRIPT_PATH
# fi
# echo ""

# echo "----------------- Process dataset has digit ------------------"
# python3 clean_digit_dataset.py PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR
# if [ $? -eq 255 ];
# then
#     echo "Cannot process dataset has digit"
#     exit 1
# else
#     echo "Done process dataset has digit "$PROCESSED_DATA_DIR
# fi
# echo ""

# echo "----------------- Process dataset has OOV ------------------"
# python3 clean_oov_dataset.py PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR
# if [ $? -eq 255 ];
# then
#     echo "Cannot process dataset has OOV"
#     exit 1
# else
#     echo "Done process dataset has OOV "$PROCESSED_DATA_DIR
# fi
# echo ""

# echo "----------------- Concatenate all clean data ------------------"
# for data in $PROCESSED_DATA_DIR/clean/*; do
#     cat $data >> $ALL_TEXTS
# done

# echo "----------------- Convert MP3 to Wav format ------------------"
# python3 convert_audios.py MP3_DIR=$MP3_DIR WAV_DIR=$WAV_DIR
# if [ $? -eq 255 ];
# then
#     echo "Cannot Convert MP3 to Wav format"
#     exit 1
# else
#     echo "Done Convert MP3 to Wav format in "$WAV_DIR
# fi
# echo ""