import numpy as np
import cv2

# loading the digits data
data = cv2.imread('images/digits.png')
gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

# Resizing each digits from 20x20 to 10x10
resized = cv2.pyrDown(gray)

cv2.imshow("Original Data", resized)

# Splitting the data into 5000 cells, each cell of size 20x20
# Resulting array: 50 * 100 * 20 * 20

arr = [np.hsplit(i, 100) for i in np.vsplit(gray, 50)]
arr = np.array(arr)
print("Resulting Shape", arr.shape)

# Splitting into training and test set
# Total: 5000,Train: 3500 images, Test: 1500

x_train = arr[:, :70].reshape(-1, 400).astype(np.float32)
x_test = arr[:, 70:100].reshape(-1, 400).astype(np.float32)
print ("Input shapes\n --> Train: {}, Test: {}".format(x_train.shape, x_test.shape))

# targets for each image

y = [0,1,2,3,4,5,6,7,8,9]

y_train = np.repeat(y, 350)[:, np.newaxis]
y_test = np.repeat(y, 150)[:, np.newaxis]
print ("Target shapes\n --> Train: {}, Test: {}".format(y_train.shape, y_test.shape))

# using K-NN (K- nearest neighbors) as the ML algorithm
classifier_knn = cv2.ml.KNearest_create()
classifier_knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
response, result, neighbours, distance = classifier_knn.findNearest(x_test, k=3)

# Testing and calculating the accuracy of knn classifier
correct = result == y_test
correct = np.count_nonzero(correct)
accuracy = correct * (100.0/result.size)
print ("Accuracy: ", accuracy)
cv2.waitKey(0)
cv2.destroyAllWindows()