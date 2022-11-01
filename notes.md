# INTRODUCTION TO ML AND AI

## What is Machine Learning and AI?

The first question people back in 1950 asked was "Can machines think?". This is where people started to simulate inteligence in machines.

And we need to understand that ml is just a branch of AI. AI is a broader term that includes ml, deep learning, natural language processing, computer vision, etc.

## What is Machine Learning?

Using machine learning as in contrast to traditional programming, we can build systems thato follow a set of rules to solve a problem. In machine learning, we can build systems based in data and answers and the system will generate the rules for us.
The goal for machine learning is to build a system that can learn from data and make predictions with the highest accuracy possible.

### Important concepts

#### Data

Data is the most important concept in machine learning. Data is the fuel for the machine learning algorithms. The more data we have, the better the model will be.

- Features: The features are the input information that we use to make predictions. The features are the columns in a table. (Information we always have).
- Labels: The labels are the output information that we want to predict. The labels are the rows in a table. (Information we want to predict).

In the first step when we train a model we will have available the features and labels. But in future steps to test the model we will only have the features and we will need to predict the labels.

### Types of Machine Learning

#### Supervised Learning

In supervised learning we have a set of features and labels. The goal is to train a model that can predict the labels based on the features.

Flow:
1. We train the model with the features and labels.
2. We pass the features to the model and the model will predict the labels.
3. Compare the predicted labels with the real labels.
4. Tweaks the model to improve the accuracy.
5. Repeat the process until we have a model with the desired accuracy.

#### Unsupervised Learning

In unsupervised learning we have a set of features but we don't have the labels. The goal is to train a model that can find patterns in the data.

#### Reinforcement Learning

In reinforcement learning we have a set of features and labels. The goal is to train a model that can predict the labels based on the features. But in this case when we test the model we won't have the labels. We will need to reward the model when it makes a good prediction and punish the model when it makes a bad prediction.

## Before entering to ML

### Probabilistic modeling

Probabilistic modeling is the application of the principles of statistics to data analysis. It was one of the earliest forms of machine learning, and it’s still widely used to this day.

### Naive Bayes algorithm

Naive Bayes is a type of machine-learning classifier based on applying Bayes’ theorem while assuming that the features in the input data are all independent (a strong, or “naive” assumption, which is where the name comes from).

### Logistic regression (logreg)

Much like Naive Bayes, logreg predates computing by a long time, yet it’s still useful to this day, thanks to its simple and versatile nature. It’s often the first thing a data scientist will try on a dataset to get a feel for the classification task at hand.

### Early neural networks

The core ideas of neural networks were investigated in toy forms as early as the 1950s, the approach took decades to get started. For a long time, the missing piece was an efficient way to train large neural networks. 

This changed in the mid-1980s, when multiple people independently rediscovered the Backpropagation algorithm— a way to train chains of parametric operations using gradient-descent optimization —and started applying it to neural networks.

### Kernel methods

Kernel methods are a group of classification algorithms, the best known of which is the support vector machine (SVM).

SVMs aim at solving classification problems by finding good decision boundaries between two sets of points belonging to two different categories. A decision boundary can be thought of as a line or surface separating your training data into two spaces corresponding to two categories. To classify new data points, you just need to check which side of the decision boundary they fall on.

SVMs proceed to find these boundaries in two steps:

1. The data is mapped to a new high-dimensional representation where the decision boundary can be expressed as a hyperplane.
2. A good decision boundary (a separation hyperplane) is computed by trying to maximize the distance between the hyperplane and the closest data points from each class, a step called maximizing the margin. This allows the boundary to generalize well to new samples outside of the training dataset

### Decision trees, random forests, and gradient boosting machines

Decision trees are flowchart-like structures that let you classify input data points or predict output values given inputs. They’re easy to visualize and interpret. in the 2000s they were often preferred to kernel methods because they were easier to train and tune.

In particular, the Random Forest algorithm introduced a robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and then ensembling their outputs

A gradient boosting machine, much like a random forest, is a machine-learning technique based on ensembling weak prediction models, generally decision trees. It uses gradient boosting, a way to improve any machine-learning model by iteratively training new models that specialize in addressing the weak points of the previous models.
Applied to decision trees, the use of the gradient boosting technique results in models that strictly outperform random forests most of the time, while having similar properties.

### Back to neural networks

In 2011, Dan Ciresan from IDSIA began to win academic image-classification competitions with GPU-trained deep neural networks—the first practical success of modern deep learning. But the watershed moment came in 2012, with the entry of Hinton’s group in the yearly large-scale image-classification challenge ImageNet.

In 2011, the top-five accuracy of the winning model, based on classical approaches to computer vision, was only 74.3%. Then, in 2012, a team led by Alex Krizhevsky and advised by Geoffrey Hinton was able to achieve a top-five accuracy of 83.6%—a significant breakthrough. The competition has been dominated by deep convolutional neural networks every year since. By 2015, the winner reached an accuracy of 96.4%, and the classification task on ImageNet was considered to be a completely solved problem.

Since 2012, deep convolutional neural networks (convnets) have become the go-to algorithm for all computer vision tasks; more generally, they work on all perceptual tasks.

## The mathematical building blocks of neural networks

### First look at a neural network

Let’s look at a concrete example of a neural network that uses the Python library Keras to learn to classify handwritten digits. The problem we’re trying to solve here is to classify grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9). We’ll use the MNIST dataset, a classic in the machine-learning community, which has been around almost as long as the field itself and has been intensively studied. It’s a set of 60,000 training images, plus 10,000 test images.

The MNIST dataset comes preloaded in Keras, in the form of a set of four Numpy arrays.

1. Loading the MNIST dataset in Keras
   
    ```python
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    ```

   train_images and train_labels form the training set, the data that the model will learn from. The model will then be tested on the test set, test_images and test_labels.

   The images are encoded as Numpy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.

   If we take a loof at the training data:

    ```python
    >>> train_images.shape
    (60000, 28, 28)
    >>> len(train_labels)
    60000
    >>> train_labels
    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
    ```

   And the test data:

    ```python
    >>> test_images.shape
    (10000, 28, 28)
    >>> len(test_labels)
    10000
    >>> test_labels
    array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
    ```

    The workflow will be as follow:

    First we’ll feed the neural network the training data, train_images and train_labels. The network will then learn to associate images and labels.

    Finally, we’ll ask the network to produce predictions for test_images, and we’ll verify if these predictions match the labels from test_labels.

2. The network architecture
   
   ```python
    from keras import models
    from keras import layers

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
   ```
   
   - Layer

      A layer is a data-processing module that you can think of as a filter for data. Some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation.

       Here, our network consists of a sequence of two Dense layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way softmax layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes.
     
3. Compilation Step
  
   To make the network ready for training, we need to pick three more things, as part of the compilation step:

    - A loss function: How the network will be able to measure how good a job it is doing on its training data, and thus how it will be able to steer itself in the right direction.
    - An optimizer: The mechanism through which the network will update itself based on the data it sees and its loss function.
    - Metrics to monitor during training and testing: Here, we’ll only care about accuracy (the fraction of the images that were correctly classified).

    ```python
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    ```
  
4. Preparing the image data
   
   Before training, we’ll preprocess the data by reshaping it into the shape the network expects and scaling it so that all values are in the [0, 1] interval. Previously, our training images, for instance, were stored in an array of shape (60000, 28, 28) of type uint8 with values in the [0, 255] interval. We transform it into a float32 array of shape (60000, 28 * 28) with values between 0 and 1.

    ```python
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    ```

5. Preparing the labels

    We also need to categorically encode the labels.
  
    ```python
    from keras.utils import to_categorical
  
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    ```

    We’re now ready to train the network, which in Keras is done via a call to the network’s fit method—we fit the model to its training data:

    ```python
    >>> network.fit(train_images, train_labels, epochs=5, batch_size=128)
    Epoch 1/5
    60000/60000 [==============================] - 9s - loss: 0.2588 - acc: 0.9254
    Epoch 2/5
    51328/60000 [=========================>....] - ETA: 1s  - loss: 0.1040 - acc: 0.9690
    ```

Two quantities are displayed during training: the loss of the network over the training data, and the accuracy of the network over the training data. We quickly reach an accuracy of 0.989 (98.9%) on the training data. Now let’s check that the model performs well on the test set, too:
  
  ```python
  >>> test_loss, test_acc = network.evaluate(test_images, test_labels)
  >>> print('test_acc:', test_acc)
  test_acc: 0.9785  
  ```

The test-set accuracy turns out to be 97.8%—that’s quite a bit lower than the training set accuracy. This gap between training accuracy and test accuracy is an example of overfitting: the fact that machine-learning models tend to perform worse on new data than on their training data. 

### Data Representation for Neural Networks

#### Tensor

At its core, a tensor is a container for data—almost always numerical data. So, it’s a container for numbers. You may be already familiar with matrices, which are 2D tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an axis).

#### Scalars (0D tensors)