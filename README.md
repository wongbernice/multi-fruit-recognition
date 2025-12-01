# 🍓🍑 Multi-Label Fruit Classification 🍌🍉

A PyTorch project that trains a convolutional neural network to detect multiple fruits in a single image using a synthetic multi-fruit dataset.

## 📌 Overview

In this work, we extend traditional image classification models to handle multi-label fruit recognition. We demonstrated the model’s capability to identify multiple fruits within a single image. We created a new multi-fruit image dataset from the Fruits-360 dataset. The dataset was filtered to include 10 fruit classes, and experiments were run to evaluate the model's performance on training and testing sets. We trained a convolutional neural network (CNN) with sigmoid outputs and binary cross-entropy loss to predict multiple labels per image. Preliminary results indicate that the model is capable of accurately recognizing multiple fruits in a single image.

Classes included: Banana, Blueberry, Cherimoya, Lemon, Lychee, Peach, Pineapple, Raspberry, Strawberry, Watermelon

Fruits-360 Dataset: https://www.kaggle.com/datasets/moltean/fruits
