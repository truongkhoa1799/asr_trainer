# Conformer_ASR

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#load-and-start-docker-image">Load and start docker image</a>
      <ul>
        <li><a href="#load-image">Load image</a></li>
        <li><a href="#start-image">Start image</a></li>
      </ul>
    </li>
    <li>
      <a href="#evaluate-and-clean-asr-datasets">Evaluate and clean ASR datasets</a>
      <ul>
        <li><a href="#evaluate-asr-datasets">Evaluate ASR datasets</a></li>
        <li><a href="#clean-asr-datasets">Clean ASR datasets</a></li>
      </ul>
    </li>
    <li><a href="#prepare-datasets">Prepare Datasets</a></li>
    <ul>
        <li><a href="#infore-415h-dataset">Infore 415H dataset</a></li>
        <li><a href="#infore-25h-dataset">Infore 25H dataset</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- LOAD AND START DOCKER IMAGE -->
## Load and Start Docker Image
### Load image
1. Enter Conformer Repo
   ```sh
   conformer
   ```
2. Load rebuild image
   ```sh
   cd docker && docker image load -i nova_asr_imgage.tar
   ```
### Start image
1. Create container
   ```sh
   docker create --gpus all -it --rm -v /mnt/8T_Disk2/khoatlv:/home/khoatlv --shm-size=12g --ulimit memlock=-1 --ulimit  stack=67108864 --device=/dev/sda --name train_asr khoatr1799/nova_asr_image
   ```
2. Start and enter container
   ```sh
   docker container start train_asr && docker exec -it train_asr bash
   ```
3. Install evironment requirements
   ```sh
   cd ASR_Nemo/docker/ && pip3 install -r requirements.txt
   ```

<!-- Evaluate ASR dataset -->
## Evaluate and Clean ASR Datasets
In this section, we perform the following steps to evaluate the ASR dataset. Then, the result will be stored and analysis to create final cleaned manifest for the next step, create training and testing manifests.

### Evaluate ASR Datasets
1. Run the following command:
  ```sh
  export PYTHONPATH=$PWD
  python3 conformer_asr/evaluation/evaluate_asr_data.py \
  --evaluation \
  --manifest_path={} \
  --dataset_name={}
  ```
2. The result is then stored in the path of field *evaluation.result_directory* with the following structure:
  ```sh
  result_directory
    - dataset_name
      - {dataset_name}_result.log
      - {dataset_name}_error.log
  ```

### Clean ASR datasets
Based on the result of the below steps, we can remove bad data in the evaluation manifest by checking which data has wer >= threshold_wer.
1. Run the following command:
  ```sh
  python3 conformer_asr/evaluation/evaluate_asr_data.py \
  --clean_evaluation_dataset \
  --manifest_path={} \
  --clean_manifest_path={} \
  --log_data_result_path={} \
  --log_data_error_path={} \
  --threshold_wer={}
  ```


<!-- PREPARE DATASETS -->
## Prepare Datasets
This section contains script for creating the *training* and *testing* manifest of datasets
### Infore 415H dataset:
  * This datasets is the VLSP 2019
  * In other to create *training* and *testing* manifest for this dataset, we:  
    1. Create manifest from the orginal manifest (*data_book_train_relocated.json*) by replace the directory of wave data in this manifest with the one we store audio (*book_relocated*)
    ```sh
    python3 conformer_asr/data/infore_datasets/prepare_infore_415h.py --create_manifest
    ```
    2. We perform the evaluation asr datasets in the below step to create *cleaned manifest*
    3. Finally, create *training* and *testing* manifest with the following command:
    ```sh
    python3 conformer_asr/data/infore_datasets/prepare_infore_415h.py --split_dataset
    ```

### Infore 25H dataset:
  * In other to create *training* and *testing* manifest for this dataset, we:  
    1. Create manifest from the list of waves and scripts by matching the value of 2 dictionary.
    ```sh
    python3 conformer_asr/data/infore_datasets/prepare_infore_25h.py --create_manifest
    ```
    2. Then, create *training* and *testing* manifest with the following command:
    ```sh
    python3 conformer_asr/data/infore_datasets/prepare_infore_25h.py --split_dataset
    ```

### VLSP 2020:
  * The training and testing manifest of this datasets are created by mapping the dictionary of scripts and audio path from traing and test set
  * Here we have 2 set datasets. Therefore, in order to create the manifest, we run the following command:
    ```sh
    python3 conformer_asr/data/prepare_vlsp2020.py --set_{nummber}
    ```
  * *number* is the value for which set we need to create training and testing manifest

### VLSP 2021:
  * The training and testing manifest of this datasets are created by mapping the dictionary of scripts and audio path from traing and test set
    ```sh
    python3 conformer_asr/data/prepare_vlsp2021.py
    ```
  * Then, we perform the evaluation of the training and testing manifest to remove error and bad audio files:
    * Training set
      ```sh
      python3 conformer_asr/evaluation/evaluate_asr_data.py \
      --evaluation \
      --manifest_path=/home/khoatlv/data/vlsp2021/manifests/vlsp2021_train_manifest.json \
      --dataset_name=vlsp2021_train
      ```

      ```sh
      python3 conformer_asr/evaluation/evaluate_asr_data.py 
      --clean_evaluation_dataset \
      --manifest_path=/home/khoatlv/data/vlsp2021/manifests/vlsp2021_train_manifest.json \
      --clean_manifest_path=/home/khoatlv/data/vlsp2021/manifests/vlsp2021_train_manifest_cleaned.json \
      --log_data_result_path=/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data/vlsp2021_train/vlsp2021_train_result.log \
      --log_data_error_path=/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data/vlsp2021_train/vlsp2021_train_error.log \
      --threshold_wer=0.2
      ```
    * Testing set
      ```sh
      python3 conformer_asr/evaluation/evaluate_asr_data.py \
      --evaluation \
      --manifest_path=/home/khoatlv/data/vlsp2021/manifests/vlsp2021_test_manifest.json \
      --dataset_name=vlsp2021_test
      ```

      ```sh
      python3 conformer_asr/evaluation/evaluate_asr_data.py 
      --clean_evaluation_dataset \
      --manifest_path=/home/khoatlv/data/vlsp2021/manifests/vlsp2021_test_manifest.json \
      --clean_manifest_path=/home/khoatlv/data/vlsp2021/manifests/vlsp2021_test_manifest_cleaned.json \
      --log_data_result_path=/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data/vlsp2021_test/vlsp2021_test_result.log \
      --log_data_error_path=/home/khoatlv/ASR_Nemo/conformer_asr/evaluation/results/ASR_data/vlsp2021_test/vlsp2021_test_error.log \
      --threshold_wer=0.2
      ```





<!-- ABOUT THE PROJECT -->
## DATASETS
### **1. Configure training datasets**
In the **training_config.yml**, we are able to configure which data we want to train by modify the boolean parameter

General Data Training:
* use_common_voice:
  * Total hours: 0.66
  * Training data:
    * Hours: 0.44
    * Numbef of audios: 417
  * Testing data:
    * Hours: 0.22
    * Numbef of audios: 199 

* use_vivos:
  * Total hours: 15.67
  * Training data:
    * Hours: 14.92
    * Numbef of audios: 11660
  * Testing data:
    * Hours: 0.75
    * Numbef of audios: 760 

* use_vlsp2020_set1
  * Total hours: 121.59
  * Training data:
    * Hours: 114.09
    * Numbef of audios: 118950
  * Testing data:
    * Hours: 7.5
    * Numbef of audios: 7500 

* use_vlsp2020_set2
  * Total hours: 87.81
  * Training data:
    * Hours: 81.8
    * Numbef of audios: 42152
  * Testing data:
    * Hours: 6.01
    * Numbef of audios: 18843 

* use_vlsp2021
  * Total hours: 23.59
  * Training data:
    * Hours: 19.84
    * Numbef of audios: 15720
  * Testing data:
    * Hours: 3.75
    * Numbef of audios: 1291 

* use_infore_25h
  * Total hours: 24.86
  * Training data:
    * Hours: 22.38
    * Numbef of audios: 13338
  * Testing data:
    * Hours: 2.48
    * Numbef of audios: 1482 

* use_infore_415h
  * Total hours: 330.73
  * Training data:
    * Hours: 297.78
    * Numbef of audios: 220517
  * Testing data:
    * Hours: 32.96
    * Numbef of audios: 24502 

Collected Data Training
* use_viettel
  * Total hours: 4.97
  * Training data:
    * Hours: 4.35
    * Numbef of audios: 14001
  * Testing data:
    * Hours: 0.62
    * Numbef of audios: 1976 

* use_viettel_assistant
  * Total hours: 0.36
  * Training data:
    * Hours: 0.3
    * Numbef of audios: 850
  * Testing data:
    * Hours: 0.06
    * Numbef of audios: 170 

* use_fpt
* use_zalo

### **2. Summary dataset**
Inorder to summarize the information of datasets, we execute the following command:
* python3
  ```sh
  export PYTHONPATH=$PWD
  python3 conformer_asr/utils.py --summarize_dataset
  ```

### **3. Create Manifest for training**
In this section, we create training manifest and testing manifest for conformer trainer. The following step will be conducted in this script:


* python3
  ```sh
  export PYTHONPATH=$PWD
  python3 conformer_asr/data/create_manifest_dataset.py
  ```
1. First, we choose which dataset is used for training in the **training_config.yml**
2. Then, based on the training configuration, we create the training manifest and testing manifest for the training phase based on the *training_manifest* and *testing_manifest* of datasets.
3. Next, with the training and testing manifest, we perform the following preprocessing steps and store with preprocessed data:
    * Remove special characters
    * Lower case
    * Strip begin and last characters
4. Finally, we extract the characters and summarize number of token in the datasets

<!-- Tokenizers -->
## TOKENIZERS
In this section, we create the tokenizers for Conformer model
1. Create Tokenizer directory by execute the following command:
    ```sh
    export PYTHONPATH=$PWD
    python3 conformer_asr/tokenizer/create_tokenizer.py
    ```
2. Then the tokenizers directory will be store in the folder **/home/khoatlv/Conformer_ASR/tokenizers** with name *tokenizers_conformer_{timestamp}*
3. Modify the attribute *training.tokenizer.tokenizer_dir* in the **training_config.yml** with new value *tokenizers_conformer_{timestamp}* created above

<!-- Training -->
## TRAINING
1. Create screen for training
  ```sh
  screen -S asr 
  docker exec -it base_nemo bash
  cd /home/khoatlv/Conformer_ASR
  export PYTHONPATH=$PWD
  ```
2. Run the following command to create cleaned training and testing manifests
  ```sh
  python3 conformer_asr/data/create_manifest_dataset.py
  ```
3. Create tokenizer
 * If we start fintune from pretrained English, we have to create new tokenizer for new datasets
    ```sh
    python3 conformer_asr/tokenizer/create_tokenizer.py
    ```
* Modify the attribute *training.tokenizer.tokenizer_dir* in the **training_config.yml** with new value *tokenizers_conformer_{timestamp}* created above
4. Run training script
  ```sh
  python3 conformer_asr/training/train_conformer.py
  ```
  
<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

truong Khoa - [@khoatruong1799](https://twitter.com/khoatruong1799) - khoa.truong1799@gmail.com

Project Link: [https://github.com/truongkhoa1799/ASR_Nemo](https://github.com/truongkhoa1799/ASR_Nemo)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
