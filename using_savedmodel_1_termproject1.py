from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPool2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tempfile import NamedTemporaryFile
from numpy import savetxt
import tensorflow
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPool2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tempfile import NamedTemporaryFile
from numpy import savetxt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn import metrics
# Using nsaved from termproject1
new_model = tensorflow.keras.models.load_model('model-2convo-new.h5')
# Taking random image from database to predic its class label
img_path = "COVID-19_Radiography_Database/COVID-19/COVID-19 (26).png"

test_image = image.load_img(img_path, target_size=(227, 227))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.

# predicting classes
images = np.vstack([test_image])
classes = new_model.predict_classes(images)

print("Predicted class of the random test image is:", classes)

# Visualizing an image under filters and max pooling
# Taking an random image from dataset to observe functions of different layers

# Gets the output from top 4 layers which includes 2 convolution layers, and 2 max pooling layers
outputs_from_layers = [layer.output for layer in new_model.layers[:4]]

visualization_model = Model(inputs=new_model.input,
                         outputs=outputs_from_layers)

Predictions = visualization_model.predict(test_image)
names_of_layers = []
# Adding names
for layer in new_model.layers[:4]:
    names_of_layers.append(layer.name)

def visualize(names_of_layers, Predictions):
    row_size = 16

    # Plotting each layer
    for l_name, activation in zip(names_of_layers, Predictions):
        number_of_features = activation.shape[-1]
        size = activation.shape[1]
        number_of_columns = number_of_features // row_size

        #initialize the grid with zeros.
        d_grid = np.zeros(((size * number_of_columns), (row_size * size)))

        for col in range(number_of_columns):
            for row in range(row_size):
                c_image = activation[0, :, :, col * row_size + row]
                c_image -= c_image.mean()
                c_image *= 64
                c_image += 128
                c_image = np.clip(c_image, 0, 255).astype('uint8')
                d_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = c_image

        scale = 1.0 / size
        plt.figure(figsize=(scale * d_grid.shape[1],
                            scale * d_grid.shape[0]))
        plt.title(l_name)
        plt.imshow(d_grid, aspect='auto', cmap='viridis')
        plt.show()

visualize(names_of_layers, Predictions)
data = np.load("covid_data")
labels = np.load("covid_labels")

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.20, stratify=labels,
                                                                      random_state=42)
preds = new_model.predict(test_data)
acc = accuracy_score(test_labels, np.round(preds)) * 100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp / (tp + fp) * 100
recall = tp / (tp + fn) * 100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2 * precision * recall / (precision + recall)))

print('\nTRAIN METRIC ----------------------')
# print('Train acc: {}'.format(np.round((histogram.history['accuracy'][-1])*100, 2)))

y_val_cat_prob = new_model.predict_proba(test_data)
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(test_labels, y_val_cat_prob)


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.savefig("roc.png")


plot_roc_curve(fpr, tpr)
print("AUC score:" + str(metrics.auc(fpr, tpr)))
