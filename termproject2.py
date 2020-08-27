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
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loading images
imagePaths = list(paths.list_images("COVID-19_Radiography_Database"))
data = []
labels = []



# Loop over the image paths
for imagePath in imagePaths:

    # Extract the class label from the filename
    name = imagePath.split(os.path.sep)[-2]

    # Loading the image, and resizing it to 227*227

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = cv2.resize(image, (227, 227))
    # Plotting a histogram before denoising
    plt.hist(r.ravel(), 256, [0, 256])
    dst = cv2.fastNlMeansDenoising(image, None, 3, 7, 21)
    image = cv2.resize(dst, (227, 227))
    image = image.astype('float32')/255


    #print(name)
    # Marking labels
    if name == 'NORMAL' or name == 'ViralPneumonia':
        label = 0
    else:
        label = 1


    # updating the data and labels lists
    data.append(image)
    labels.append(label)

# Converting the data and labels to NumPy arrays
data = np.array(data)


# Saving data and labels to output files to use directly without loading all the images
# when we use saved model and to make predictions.
outfile = NamedTemporaryFile()
np.save(outfile, data)
labels = np.array(labels)

outfile2 = NamedTemporaryFile()
np.save(outfile2, labels)

partition the data into training and testing splits using 80% of
the data for training and the remaining 20% for testing
(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.20, stratify=labels,
                                                                      random_state=42)

print("completed training and testing dataset split")


# Initialize the training data augmentation object
# trainAug = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.3,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         vertical_flip=True, fill_mode="nearest")

#Initializing all variables
img_dims = 227
epochs = 2
batch_size = 5

inputs = Input(shape=(img_dims, img_dims, 3))

# First conv block
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(227, 227, 3)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Second Conv block
model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


# Fully connected layer and output layer
model.add(Flatten())
model.add(Dense(units=128 , activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=64 , activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=1, activation='sigmoid'))
print("all layers done")


# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("compilation done")
# Callbacks
checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
# Fitting the model along with data augmentation
histogram = model.fit_generator(aug.flow(train_data, train_labels, batch_size=64),
                                validation_data=(test_data, test_labels), steps_per_epoch=len(train_data),
                                epochs=10)

# Plotting auucuracy and loss curves and saving the figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(histogram.history[met])
    ax[i].plot(histogram.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

plt.show()
plt.savefig("acc_loss.png")

#Saving the model after training
model.save("model-4convolayers-new.h5")

#Predictions using test data
preds = model.predict(test_data)

acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()
#printing Accuracy, Confusion matrix, Precision, Recall, F1-Score, Training accuracy
print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('\nTRAIN METRIC ----------------------')
print('Train acc: {}'.format(np.round((histogram.history['accuracy'][-1])*100, 2)))

# Predicting probabilities for ROC Curve
y_val_prob=model.predict_proba(test_data)
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(test_labels, y_val_prob)
# Plotting, saving the ROC Curveand printing the AUC Score
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.savefig("roc.png")


plot_roc_curve(fpr, tpr)

print("AUC score:" +str(metrics.auc(fpr, tpr)))