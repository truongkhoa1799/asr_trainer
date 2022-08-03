# Conformer_ASR

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#load-and-start-docker-image">Config selected data training</a>
      <ul>
        <li><a href="#load-image">Load image</a></li>
        <li><a href="#Start-image">Start image</a></li>
      </ul>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- LOAD AND START DOCKER IMAGE -->
## LOAD AND START DOCKER IMAGE
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
   docker create --gpus all -it --rm -v /mnt/8T_Disk2/khoatlv:/home/khoatlv --shm-size=12g --ulimit memlock=-1 --ulimit  stack=67108864 --device=/dev/sda --name base_nemo nova/asr
   ```
2. Start and enter container
   ```sh
   docker container start base_nemo && docker exec -it base_nemo bash
   ```
3. Install evironment requirements
   ```sh
   cd Conformer_ASR/docker/ && pip3 install -r requirements.txt
   ```

<!-- Evaluate ASR dataset -->
## EVALUATE AND CLEAN ASR DATASETS
In this section, we perform the following steps to evaluate the ASR dataset. Then, the result will be stored and analysis to create final cleaned manifest for the next step, create training and testing manifests.

### **1. Evaluate ASR datasets**

1. First, we choose which datasets we want to evaluate by modify the value of the field *evaluation.dataset* in **training_config.yml** to the manifest of this datasets.
2. Change the field *evaluation.resutl.dataset_name* to the datasets name for logging reuslt
3. Run the following command:
  ```sh
  export PYTHONPATH=$PWD
  python3 conformer_asr/evaluation/evaluate_asr_data.py --evaluation
  ```

### **2. Clean ASR datasets**
Based on the result of the below steps, we can remove bad data in the evaluation manifest by checking which data has wer >= threshold_wer defined in the **training_config.yml**.
1. Run the following command:
  ```sh
  python3 conformer_asr/evaluation/evaluate_asr_data.py --clean_evaluation_dataset
  ```
2. The clean datasets manifest will be store at the same directory with evaluation manifest. However, the name will be added with **_clean** at the end of evaluation manifest name.

<!-- ABOUT THE PROJECT -->
## PREPARE DATASETS
This section contains script for creating the *training* and *testing* manifest of datasets
1. Infore 415H dataset:
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

2. Infore 25H dataset:
  * In other to create *training* and *testing* manifest for this dataset, we:  
    1. Create manifest from the list of waves and scripts by matching the value of 2 dictionary.
    ```sh
    python3 conformer_asr/data/infore_datasets/prepare_infore_25h.py --create_manifest
    ```
    2. Then, create *training* and *testing* manifest with the following command:
    ```sh
    python3 conformer_asr/data/infore_datasets/prepare_infore_25h.py --split_dataset
    ```



<!-- ABOUT THE PROJECT -->
## DATASETS
### **1. Configure training datasets**
In the **training_config.yml**, we are able to configure which data we want to train by modify the boolean parameter

General Data Training:
* use_common_voice
* use_vivos
* use_vlsp2020_set1
* use_vlsp2020_set2
* use_infore_25h
* use_infore_415h

Collected Data Training
* use_viettel
* use_viettel_assistant
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
    python3 conformer_asr/tokenizer/create_manifest_dataset.py
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
  

<p align="right">(<a href="#top">back to top</a>)</p>
* Next-url
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]




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

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

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
