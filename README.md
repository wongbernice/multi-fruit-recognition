# 🍓🍑 Multi-Label Fruit Classification 🍌🍉

A PyTorch project that trains a convolutional neural network to detect multiple fruits in a single image using a synthetic multi-fruit dataset.

## 📌 Overview

In this work, we extend traditional image classification models to handle multi-label fruit recognition. We demonstrated the model’s capability to identify multiple fruits within a single image. We created a new multi-fruit image dataset from the Fruits-360 dataset. The dataset was filtered to include 10 fruit classes, and experiments were run to evaluate the model's performance on training and testing sets. We trained a convolutional neural network (CNN) with sigmoid outputs and binary cross-entropy loss to predict multiple labels per image. Preliminary results indicate that the model is capable of accurately recognizing multiple fruits in a single image.

## 📂 Dataset
- **Source**: [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (100 x 100 images)
- **Classes**: Banana, Blueberry, Cherimoya, Lemon, Lychee, Peach, Pineapple, Raspberry, Strawberry, Watermelon
- **Multi-fruit Images**: Each image contains 2-6 fruits combined from different classes
- **Training / Testing**: 500 / 200 images
- **Preparation Scripts**:
  - `filter_dataset.py` → selects the desired fruit classes from the original dataset
  - `create_multi_dataset.py` → combines the single-fruit images into multi-fruit images
  - `MultiFruitDataset.py ` → PyTorch Dataset class for loading images and labels
- **Download Link**: [Multi-Fruit Dataset](https://ucf-my.sharepoint.com/:f:/g/personal/yi951759_ucf_edu/IgBanlWhyFIVSbRYbQ9pFA95AUeTkKw6oflSXTc69MgPxxY?e=7xY3X1)

## 🛠️ Installation
Install required libraries:  
`pip install torch torchvision numpy matplotlib scikit-learn opencv-python pillow`

## 🖥️ Usage
1. Prepare Dataset
   -  Create your own dataset:  
      `python filter_dataset.py`  
      `python create_multi_dataset.py`
   -  OR download the [dataset](https://ucf-my.sharepoint.com/:f:/g/personal/yi951759_ucf_edu/IgBanlWhyFIVSbRYbQ9pFA95AUeTkKw6oflSXTc69MgPxxY?e=7xY3X1) we made. 
2. Train and Evaluate Model  
   `python train_evaluate_cnn.py --learning_rate 0.001 --num_epochs 60 --batch_size 16`
