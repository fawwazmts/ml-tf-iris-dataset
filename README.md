# ML Model with Tensorflow and Iris Dataset from scikit-learn



## Description About Project

This experiment will build an ML model to determine the types of iris (Setosa, Versicolour, and Virginica) based on sepal length, sepal width, petal length and petal width. The model is a linear regression model.

This repository stores how to build models. The data used to build the model is also available in this repository.

## Infrastructure for Built Model

### Libraries

Model Machine Learning for Aksacarma built with Jupyter Notebook on Google Colab. This Jupyter Notebook also can run in local. For development, the following are libraries which needs to be installed.

```
numpy==1.25.2
pandas==2.0.3
tensorflow==2.15.0
seaborn==0.13.1
sklearn==1.2.2
matplotlib==3.7.1
```

### Dataset

This model was built utilizing a dataset from [scikit-learn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

## How to Use This Project

For using this project you must clone this machine learning project with following command.

`git clone https://github.com/fawwazmts/ml-iris-dataset.git`

### How to Build Models

Complete steps on how to build the model are available on [ml_tf_model_iris_dataset_sklearn.ipynb]https://github.com/fawwazmts/ml-tf-iris-dataset/blob/main/ml_model_iris_tf_dataset_sklearn.ipynb). A summary of the steps is as follows.

1. Get Iris Dataset from [scikit-learn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) using `load_iris()`
2. Split data to train, validation, and test dataset.
3. Train model with data.
