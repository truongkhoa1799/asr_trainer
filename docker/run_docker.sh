echo "Build docker image"
sudo docker build -t nova/asr -f Dockerfile .

docker create --gpus all -it --rm -v /mnt/8T_Disk2/khoatlv:/home/khoatlv --shm-size=12g --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/sda --name base_nemo nova/asr
