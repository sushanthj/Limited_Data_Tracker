#!/bin/bash

# Download all dependencies
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Clone the 2D-to-3D repository
git clone https://github.com/sushanthj/2D_to_3D.git

python3 -m pip install gdown

DATA_LINK="https://drive.google.com/uc?id=1g5n_QAqxdvXQlnvS6piF9ZYHhm-5Cy4D"
CHECKPOINT_LINK="https://drive.google.com/uc?id=14k4eoYypbD-Ci-fGlgzJw0qvkQCC38lq"
TRAIN_DATA_LINK="https://drive.google.com/uc?id=1NwKpmasyGj_y593nFAzi5fdLDLc9pg-9"

RELATIVE_PATH="classification/checkpoints/"

# Get the current directory
CURRENT_DIR=$(pwd)

# Construct the absolute path of the new folder
FOLDER="$CURRENT_DIR/$RELATIVE_PATH"

# Download the file using gdown
gdown $CHECKPOINT_LINK -O checkpoint.pth
gdown $DATA_LINK -O data.zip
gdown $TRAIN_DATA_LINK -O train_data.zip

# Move the unzipped file to the specific folder
mkdir $FOLDER
mv checkpoint.pth $FOLDER

# unzip the data files (for inference)
unzip -qo 'data.zip'

# unzip the training data files (for training)
unzip -qo 'train_data.zip' -d 'classification/'

rm data.zip
rm train_data.zip
