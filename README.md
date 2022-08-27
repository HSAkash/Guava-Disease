# Guava-Disease


This model predicts guava diseases.This model prediction accuracy is `98.49%`(test data) && `98%` accuracy (validation_data)
## Data preprocessing
The data_augmentation model uses for dataset preprocessing.
* Flip (horizontal)
* Roation (0.2)
* Zoom (0.2)
* Height (0.2)
* Width (0.2)
* Rescaling (0-255)-(0-1)

## Images
![screenshot](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/original.png)

## After augmetation
![augmetation_image](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/augmented_image.png)

## CNN model
CNN model is used to train the network.<br>
Layer parameters:<br>
* Input size (224,224,3)
* Conv2D with 64 filters
* Conv2D with 64 filters
* MaxPool2D (pool_size=2)
* Conv2D with 64 filters
* MaxPool2D (pool_size=2)
* GlobalAveragePooling2D
* Output

### Compile model
Compile the model with the following options:
* Loss function (categorical_crossentropy)
* optimizer (Adam lr=0.001)
* metrics (accuracy)

### Fit model
Then fit the model with the following parameters:
* train_data
* epochs (600)
* validation_data (test data)
* validation_steps (len of test_data)


#### 0.Input image
![layer_0](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/test_image.png)
#### 1.Conv2D with 64 filters (output)
![layer_0](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/layer_0.png)
#### 2.Conv2D with 64 filters (output)
![layer_2](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/layer_1.png)
#### 3.MaxPool2D (pool_size=2) (output)
![layer_1](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/layer_2.png)
#### 4.Conv2D with 64 filters (output)
![layer_2](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/layer_3.png)
#### 5.MaxPool2D (pool_size=2) (output)
![layer_3](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/layer_4.png)
#### 6.Prediction (final output)
![prediction](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/predict.png)

## Confusion Matrix
![confusion_matrix](https://github.com/HSAkash/Guava-Disease/raw/main/related_images/confusion_matrix.png)




# Requirements
* matplotlib 3.5.2
* numpy 1.23.1
* Pillow 9.2.0
* scikit-learn 1.1.1
* scipy 1.8.1
* tensorflow 2.9.1


# Demo
Here is how to run the guava disease program using the following command line.<br>
```bash
python guava.py
```

# Directories
<pre>
│  guava.py
│
├─env
├─Guava Disease Dataset
|   ├─train
|   ├─test
|   ├─val
|
</pre>

# Reference
* [Tensorflow](https://www.tensorflow.org/)
* [data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

# Links (dataset & code)
* [Kaggle-Guava Disease](https://www.kaggle.com/datasets/omkarmanohardalvi/guava-disease-dataset-4-types)
* [Kaggle-code](https://www.kaggle.com/code/hsakash/guava-disease-test-data-98-49-valid-data-98)


# Author
HSAkash
* [Facebook](https://www.facebook.com/hemel.akash.7/)
* [Kaggle](https://www.kaggle.com/hsakash)


