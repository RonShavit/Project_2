**To set up:**

>to create a new virtual environment run

>>`python -m venv venv`

>>`source venv/bin/activate` (on linux/MacOs) /
>>`source venv/Scripts/activate` (on Microsoft - git bash) /
>>`venv\Scripts\Activate.ps1` (on Microsoft - Powershell)



>to install requirements run

>>`pip install -r requirements.txt`

>from [https://drive.google.com/drive/folders/1mzN32ErDmoHCnQNiMRKr8Ldm1z1EAhwY?hl=he](https://drive.google.com/drive/folders/1mzN32ErDmoHCnQNiMRKr8Ldm1z1EAhwY?hl=he) download the following files:

>>experiment_base_optimized_finetuned.zip

>>experiment_base_optimized.zip

>>synthetic_dataset_color.zip

>>synthetic_dataset_extra1.zip

>>synthetic_dataset_v1.zip

>>test_data_set.zip

>>real_data_pack.zip

>

>run (to unzip the files):

>>`unzip experiment_base_optimized_finetuned.zip`

>>`unzip experiment_base_optimized.zip`

>>`unzip synthetic_dataset_color.zip`

>>`unzip synthetic_dataset_extra1.zip`

>>`unzip synthetic_dataset_v1.zip`

>>`unzip test_data_set.zip`

>>`unzip real_data_pack.zip`

>**The final setup folder should look like:**

    |
    |-real_data_pack
        |
        |-images
        |-gt.cvs
    |
    |-test_data_set
        |
        |-images
        |-gt.cvs
        |
    |-synthetic_dataset_v1
    |
        |-images
        |-gt.cvs
    |
    |-synthetic_dataset_color
    |
        |-images
        |-gt.cvs
    |
    |-synthetic_dataset_extra1
    |
        |-images
        |-gt.cvs
    |
    |-experiment_base_optimized
        |
        |-multihead_model.pth - (these are the weights for the Zero-shot [not fine tuned] model)
    |
    |-experiment_base_optimized_finetuned
        |
        |-finetuned_multihead.pth - (these are the weights for the fine-tuned model)
    |
    |-chess_dataset.py
    |
    |-fine_tune_multihead_triplet_v2.py
    |
    |-gray_bar_detection.py
    |
    |-predict_multihead_triplet.py
    |
    |-preprocess_mix.py
    |
    |-preprocess.py
    |
    |-process_real_games.py
    |
    |-split_board_v3.py
    |
    |-train_multihead_triplet.py
    |
    |-README.md





**To train model run:**

`python preprocess_mix.py`

`python train_multihead_triplet.py --epochs 15 --batch 256 --output experiment_base_optimized`  (NOTE : this will overwrite the weights saved in `experiment_base_optimized\multihead_model.pth`. To preserve the weights, move or rename `multihead_model.pth` bedore running this line)

`python fine_tune_multihead_triplet_v2.py --source experiment_base_optimized` (NOTE : this will overwrite the weights saved in `experiment_base_optimized_finetuned\finetuned_multihead.pth`. To preserve the weights, move or rename `finetuned_multihead.pth` bedore running this line)

**To predict a single numpy.ndarry image:**

>add the import line `from predict_multihead_triplet import predict_board`

>with "image" as a ndarray file run `predict_board(image)` to predict the board in "image"

>the return value is a torch.tensor

**To predict an image file run:**

`python predict_multihead_triplet.py --path *your image path*`

>this will print the preditction to the console

**To test accuracy run:**

`python predict_multihead_triplet.py --test 1`

>this will print accuracy test results preformed on real_data_set/test

**To use zero shot model (for predicting an image file or testing accuracy)**

>for predicting an image file run:

`python predict_multihead_triplet.py --path *your image path* --zero_shot 1`

>for testing accuracy run:

`python predict_multihead_triplet.py --test 1 --zero_shot 1`

**To plot confusion matrices**

add `--plot_path *output_path.jpg*` when running `predict_multiheadtriplt.py`. The matrices will be plotted to the given path (both piece type and color confusion matrices will be plotted, along with their respective accuracies)

