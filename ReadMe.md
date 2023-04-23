# Flower Recognition with AlexNet

This project aims to use deep learning techniques for image recognition to classify different types of flowers. The model was implemented using the PyTorch library and the AlexNet architecture, which has been shown to be effective in image classification tasks. The network consists of five convolutional layers and three fully connected layers, with ReLU activation and dropout regularization to prevent overfitting.

## Dataset

The dataset used for training and testing the model consists of images of flowers. The classes used were taken from the [Flower Recognition Challenge dataset on Kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition), which includes five different types of flowers: daisy, dandelion, rose, sunflower, and tulip.

Here are some sample images of the different types of flowers::

<div>
    <a href="https://en.wikipedia.org/wiki/Daisy">
        <img src="https://images.pexels.com/photos/67857/daisy-flower-spring-marguerite-67857.jpeg?cs=srgb&dl=pexels-pixabay-67857.jpg&fm=jpg" width="200"/>
    </a>
    <a href="https://en.wikipedia.org/wiki/Dandelion">
        <img src="https://cdn.britannica.com/44/5644-050-F793FA67/dandelion-head-flowers.jpg" width="200"/>
    </a>
    <a href="https://en.wikipedia.org/wiki/Rose">
        <img src="https://images.unsplash.com/photo-1560717789-0ac7c58ac90a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NXx8ZGFpc3l8ZW58MHx8MHx8&w=1000&q=80" width="200"/>
    </a>
    <a href="https://en.wikipedia.org/wiki/Sunflower">
        <img src="https://images.unsplash.com/photo-1616156194103-ab9d45e33915?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8MTJ8fHxlbnwwfHx8fA%3D%3D&w=1000&q=80" width="200"/>
    </a>
    <a href="https://en.wikipedia.org/wiki/Tulip">
        <img src="https://media.istockphoto.com/id/1393708668/photo/tulips.jpg?b=1&s=170667a&w=0&k=20&c=X2OwUy-brf374FihO8KcCj0-TbJrZxtpti3d7SymhHw=" width="200"/>
    </a>
</div>

## Methodology

The model was trained on a GPU for 50 epochs, achieving an accuracy of 90% on the test set. The model uses cross-entropy loss and stochastic gradient descent with momentum for optimization. Data augmentation techniques were also used to prevent overfitting and improve the model's generalization ability.

## Results

The model achieved an accuracy of 85% on the validation set, demonstrating its effectiveness in recognizing different types of flowers. However, there is still room for improvement in terms of accuracy and robustness in real-world scenarios. Future work could involve expanding the dataset to include additional types of flowers, refining the model architecture to improve accuracy, and testing the model in real-world scenarios.

## Train New Model

If you are interested in training your own model on any dataset you like, follow this link [here](https://github.com/your_username/Flower-Recognition/blob/main/train.py), and view the instructions on how to run the python script.

**Note**: Must be in the 'main' directory to run the python script titled ['train.py'](https://github.com/spencergoldberg1/FlowerDetection/tree/develop/Script).

## Conclusion

Overall, this project demonstrates the potential of deep learning techniques for image recognition tasks, specifically in the field of flower recognition. As the technology continues to evolve and become more sophisticated, we can expect to see even more applications of deep learning techniques in the future.

## Acknowledgments

Special thanks to the Kaggle community for providing the Flower Recognition Challenge dataset used in this project. Additionally, thanks to the PyTorch team for creating a powerful and easy-to-use deep learning library.

## About the Author

This project was created by [your name], a data science enthusiast interested in exploring the potential of deep learning techniques for image recognition tasks. Feel free to visit my [GitHub profile](https://github.com/spencergoldberg1) to see more of my work.

