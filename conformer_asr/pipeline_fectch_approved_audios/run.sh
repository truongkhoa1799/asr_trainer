echo "-------------------- Clone approved audios --------------------"

BASE_DIR=$(pwd)
UTILS_DIR=$BASE_DIR"/utils"
APPROVED_AUDIO_PATH=$BASE_DIR"/data/audio"
TRANSFORMED_AUDIO_PATH=$BASE_DIR"/data/transformed_audios"

echo "-------------------- Install sshpas --------------------"
echo apt-get update && apt-get install sshpass
echo ""

for directory in $APPROVED_AUDIO_PATH $UTILS_DIR $TRANSFORMED_AUDIO_PATH; do
    if [ -d $directory ]; then
        echo "Folder $directory is already exist"
    else
        echo "Create folder $directory"
        mkdir -p $directory
    fi
done

SENTENCE_PATH=$UTILS_DIR"/sentence_dict.pickle"
ACCESS_TOKEN_PATH=$UTILS_DIR"/access_token.txt"


echo "-------------------- Clone audio from user server --------------------"
sshpass -p "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" scp -P 2226 -r nova@192.168.1.101:/home/nova/smart_speaker/storage/user_record/approved/* $APPROVED_AUDIO_PATH
for user in $APPROVED_AUDIO_PATH/* ; do
    num_audios=$(ls $user | wc -l)
    echo "$user     $num_audios"
done
echo ""

echo "-------------------- Get access tokens --------------------"
python3 get_access_token.py ACCESS_TOKEN_PATH=$ACCESS_TOKEN_PATH
if [ $? -eq 255 ];
then
    echo "Cannot get access token"
    exit 1
else
    echo "Create acess token at "$ACCESS_TOKEN_PATH
fi
echo ""

echo "-------------------- Create sentence dictionary --------------------"
python3 get_all_scripts.py ACCESS_TOKEN_PATH=$ACCESS_TOKEN_PATH SENTENCE_PATH=$SENTENCE_PATH
if [ $? -eq 255 ];
then
    echo "Cannot create a sentence dictionary"
    exit 1
else
    echo "Create a sentence dictionary at "$SENTENCE_PATH
fi
echo ""

echo "-------------------- Transform audios to processed audio and text --------------------"
python3 transform_audios.py APPROVED_AUDIO_PATH=$APPROVED_AUDIO_PATH TRANSFORMED_AUDIO_PATH=$TRANSFORMED_AUDIO_PATH SENTENCE_PATH=$SENTENCE_PATH
if [ $? -eq 255 ];
then
    echo "Cannot audios to processed audio and text"
    exit 1
else
    echo "Transform audios to processed audio and text "$TRANSFORMED_AUDIO_PATH
fi