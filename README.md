# Digit Recognition with KNN

This project uses the K-Nearest Neighbors (KNN) algorithm to recognize digits from an image. The image data is preprocessed and then split into a training set and a test set. The KNN algorithm is then trained on the training set and used to make predictions on the test set. The accuracy of the model is calculated and displayed.

## Files

- `basic_knn.py`: This is the main file that contains the code for loading the data, preprocessing it, splitting it into training and test sets, training the KNN model, and testing it.
- `preprocess.py`: This file contains functions for preprocessing the image data, such as resizing the images and drawing a square around the digits.
- `evaluation.py`: This file contains code for evaluating the performance of the KNN model on a test image.

## Dependencies

- numpy
- cv2 (OpenCV)

## Usage

To run the project, simply run the `basic_knn.py` file. This will load the data, train the KNN model, and test it on the test set. The accuracy of the model will be printed to the console.

To evaluate the model on a test image, run the `evaluation.py` file. This will load a test image, preprocess it, and use the KNN model to make predictions on the digits in the image. The predicted digits will be displayed on the image.

## Results

The KNN model achieved an accuracy of XX% on the test set. (Note: The actual accuracy will depend on the specific data and parameters used.)

## Future Work

- Tune the parameters of the KNN model to improve its accuracy.
- Compare the performance of the KNN model to other machine learning algorithms.
- Use a larger and more diverse dataset to train and test the model.