# MedSAM
This is the modified repository for MedSAM: Segment Anything in Medical Images. 


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/joaquin2c/MedSAM_LITS.git`
4. Enter the MedSAM_LITS folder and run `pip install -e .`
5. Run `pip install albumentations`


## Dataset
The images and masks must be save in this way:

    Dataset/
    ├── images
    ├── masks
    ├── ├── 0

Only the "masks" folder have to contain the data inside the folder "0"

## Checkpoint
Download the [model checkpoint](https://drive.google.com/file/d/1SXpC9adkh9Wzjn0F7IoMKqq_eUc57U4Z/view?usp=sharing)

## Inference
In the *medsam_inference.ipynb* notebook you can test with a single image.  

## Validation of model
Use the command line to get the results of the model for multiple data:
```bash
python val_MedSAM_WPM.py --path DATASET_FOLDER_PATH--out_path RESULTS_FOLDER_PATH --check_path CHECKPOINT_PATH--name EXPERIMENT_NAME
```


