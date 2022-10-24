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

#### Naive Bayes algorithm

Naive Bayes is a type of machine-learning classifier based on applying Bayes’ theorem while assuming that the features in the input data are all independent (a strong, or “naive” assumption, which is where the name comes from).

#### Logistic regression (logreg)

Much like Naive Bayes, logreg predates computing by a long time, yet it’s still useful to this day, thanks to its simple and versatile nature. It’s often the first thing a data scientist will try on a dataset to get a feel for the classification task at hand.

#### Early neural networks

The core ideas of neural networks were investigated in toy forms as early as the 1950s, the approach took decades to get started. For a long time, the missing piece was an efficient way to train large neural networks. 

This changed in the mid-1980s, when multiple people independently rediscovered the Backpropagation algorithm— a way to train chains of parametric operations using gradient-descent optimization —and started applying it to neural networks.

#### Kernel methods

Kernel methods are a group of classification algorithms, the best known of which is the support vector machine (SVM).

SVMs aim at solving classification problems by finding good decision boundaries between two sets of points belonging to two different categories. A decision boundary can be thought of as a line or surface separating your training data into two spaces corresponding to two categories. To classify new data points, you just need to check which side of the decision boundary they fall on.