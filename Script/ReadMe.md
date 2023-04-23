# AlexNet for Image Classification

Facial Recognition with AlexNet

This script trains an AlexNet model on a custom image dataset and can be used to classify new images. The model is trained to recognize emotions and distinguish between happy and sad facial expressions.

## Background

Facial recognition has become increasingly important in various fields, from commerce to marketing, to identify human emotions and reactions to certain products or stimuli. By using deep learning techniques such as image recognition, companies can gain insight into how customers react to their products, and adjust their marketing strategies accordingly.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- requests
- tqdm

## Getting Started

1. Clone the repository

    ```
    git clone https://github.com/[username]/AlexNet-for-Image-Classification.git
    cd AlexNet-for-Image-Classification
    ```

2. Install the required packages

    ```
    pip install -r requirements.txt
    ```

3. Prepare your custom dataset by creating a directory containing your training data. The directory structure should be as follows:

    ```
    - train/
        - class_1/
            - image_1.jpg
            - image_2.jpg
            ...
        - class_2/
            - image_1.jpg
            - image_2.jpg
            ...
        ...
    ```

4. Train the model using the following command:

    ```
    python alexnet.py --train <path_to_dataset>
    ```

5. Once the model is trained, classify new images using the following command:

    ```
    python alexnet.py --classify <path_to_image>
    ```

## Options

- `--train`: Train the model using the specified dataset.
- `--classify`: Classify a new image using the trained model.

