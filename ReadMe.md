# Flower Recognition with AlexNet

This project aims to use deep learning techniques for image recognition to classify different types of flowers. The model was implemented using the PyTorch library and the AlexNet architecture, which has been shown to be effective in image classification tasks. The network consists of five convolutional layers and three fully connected layers, with ReLU activation and dropout regularization to prevent overfitting.

## Dataset

The dataset used for training and testing the model consists of images of flowers. The classes used were taken from the [Flower Recognition Challenge dataset on Kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition), which includes five different types of flowers: daisy, dandelion, rose, sunflower, and tulip.

Here are some sample images of the different types of flowers:

[![Daisy](https://images.pexels.com/photos/67857/daisy-flower-spring-marguerite-67857.jpeg?cs=srgb&dl=pexels-pixabay-67857.jpg&fm=jpg)](https://en.wikipedia.org/wiki/Daisy)
[![Dandelion](/images/dandelion.jpg)](https://en.wikipedia.org/wiki/Dandelion)
[![Rose](/images/rose.jpg)](https://en.wikipedia.org/wiki/Rose)
[![Sunflower](/images/sunflower.jpg)](https://en.wikipedia.org/wiki/Sunflower)
[![Tulip](/images/tulip.jpg)](https://en.wikipedia.org/wiki/Tulip)

## Methodology

The model was trained on a GPU for 50 epochs, achieving an accuracy of 90% on the test set. The model uses cross-entropy loss and stochastic gradient descent with momentum for optimization. Data augmentation techniques were also used to prevent overfitting and improve the model's generalization ability.

## Results

The model achieved an accuracy of 85% on the validation set, demonstrating its effectiveness in recognizing different types of flowers. However, there is still room for improvement in terms of accuracy and robustness in real-world scenarios. Future work could involve expanding the dataset to include additional types of flowers, refining the model architecture to improve accuracy, and testing the model in real-world scenarios.

## Train New Model

If you are interested in training your own model on any dataset you like, follow this link [here](https://github.com/your_username/Flower-Recognition/blob/main/train.py), and view the instructions on how to run the python script.

**Note**: Must be in the 'main' directory to run the python script titled ['train.py'](https://github.com/your_username/Flower-Recognition/blob/main/train.py).

## Conclusion

Overall, this project demonstrates the potential of deep learning techniques for image recognition tasks, specifically in the field of flower recognition. As the technology continues to evolve and become more sophisticated, we can expect to see even more applications of deep learning techniques in the future.

## Acknowledgments

Special thanks to the Kaggle community for providing the Flower Recognition Challenge dataset used in this project. Additionally, thanks to the PyTorch team for creating a powerful and easy-to-use deep learning library.

## About the Author

This project was created by [your name], a data science enthusiast interested in exploring the potential of deep learning techniques for image recognition tasks. Feel free to visit my [GitHub profile](https://github.com/spencergoldberg1) to see more of my work.

