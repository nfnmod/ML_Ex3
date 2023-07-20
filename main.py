import math
import os
from pathlib import Path
import cv2
import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image
from keras import Input, Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.callbacks import History
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split

IMAGES_PATH = "./jpg/"
PATH_GRAPHS = "./graphs"
RELU = "relu"
SOFTMAX = "softmax"
CAT_CROSS_EN = "categorical_crossentropy"
ACC = "accuracy"
TRAIN_VAL_SPLIT = 1 / 3
NUM_CATEGORIES = 102

mat = scipy.io.loadmat('./imagelabels.mat')

labels = mat['labels'][0]

VGG = VGG19(include_top=False, classes=102)
VGG.trainable = False

DENSE_UNITS = [32, 64, 128]
BATCH_SIZES = [32, 64, 128]
EPOCHS = [25, 50, 75, 100]
INPUT_SIZE = [112]


def preprocess(size):
    images = []
    for img_file in os.listdir(IMAGES_PATH):
        # load an image from file
        image = load_img(IMAGES_PATH + img_file, target_size=(size, size))

        # convert the image pixels to a numpy array
        image = img_to_array(image)

        # reshape data for the model
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))

        # prepare the image for the VGG model
        image = preprocess_input(image)

        images.append(image)

    return np.array(images)


def train_val_split(X_train, y_train):
    len_train = len(X_train)
    indices = np.arange(len_train)
    np.random.shuffle(indices)

    val_allocation = int(len_train * TRAIN_VAL_SPLIT)

    img_train = np.array(X_train)[indices[val_allocation:]]
    img_val = np.array(X_train)[indices[:val_allocation]]

    label_train = np.array(y_train)[indices[val_allocation:]]
    label_val = np.array(y_train)[indices[:val_allocation]]

    return img_train, img_val, label_train, label_val


def one_hot_encode_labels():
    encoded_labels = []
    for label in labels:
        encoded_label = np.zeros(NUM_CATEGORIES)
        encoded_label[label - 1] = 1
        encoded_labels.append(encoded_label)
    return encoded_labels


def create_VGG_classifier(input_size, dense_units, lr=0.001):
    inp = Input((input_size, input_size, 3))

    vgg_res = VGG(inp)

    flat = Flatten()(vgg_res)

    dense = Dense(dense_units, activation='relu')(flat)

    dense = Dense(dense_units, activation='relu')(dense)

    output = Dense(NUM_CATEGORIES, activation='softmax')(dense)

    model = Model(inp, output)

    model.compile(optimizer=Adam(lr), loss=CAT_CROSS_EN, metrics=[ACC])

    return model


def loss_acc_graphs(log, hyper_paramters, log2=None):
    """
    Plots the loss and accuracy graphs based on the model's log.

    Args:
        log (keras.callbacks.History): The model's log.
        hyper_paramters (str): Experiment's identification through hyper-parameters

    Returns:
        pd.DataFrame: A dataframe containing the loss and accuracy values
            of the last epoch for the training and validation sets.
    """

    # Plot loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(log.history['loss']) + 1)),
        y=log.history['loss'] if log2 is None else (np.array(log.history['loss']) + np.array(log2.history['loss'])) / 2,
        mode='lines',
        name='train'
    ))
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(log.history['val_loss']) + 1)),
        y=log.history['val_loss'] if log2 is None else (np.array(log.history['val_loss']) + np.array(
            log2.history['val_loss'])) / 2,
        mode='lines',
        name='validation'
    ))
    fig_loss.update_layout(
        title='Model Loss',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        legend=dict(x=0, y=1, traceorder='normal')
    )
    pio.write_image(fig_loss, f'{PATH_GRAPHS}/loss_{hyper_paramters}.png')

    # Plot accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=list(range(1, len(log.history['accuracy']) + 1)),
        y=log.history['accuracy'] if log2 is None else (np.array(log.history['accuracy']) + np.array(
            log2.history['accuracy'])) / 2,
        mode='lines',
        name='train'
    ))
    fig_acc.add_trace(go.Scatter(
        x=list(range(1, len(log.history['val_accuracy']) + 1)),
        y=log.history['val_accuracy'] if log2 is None else (np.array(log.history['val_accuracy']) + np.array(
            log2.history['val_accuracy'])) / 2,
        mode='lines',
        name='validation'
    ))
    fig_acc.update_layout(
        title='Model Accuracy',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Accuracy'),
        legend=dict(x=0, y=1, traceorder='normal')
    )
    pio.write_image(fig_acc, f'{PATH_GRAPHS}/accuracy_{hyper_paramters}.png')


def run_VGG():
    results = pd.DataFrame(columns=["input size", "batch size", "epochs", "dense units",
                                    "train loss 1", "train loss 2", "train acc 1", "train acc 2",
                                    "val loss 1", "val loss 2", "val acc 1", "val acc 2",
                                    "test loss 1", "test loss 2", "test acc 1", "test acc 2"])
    encoded_labels = one_hot_encode_labels()
    for input_size in INPUT_SIZE:
        images = preprocess(input_size)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(images, encoded_labels, test_size=0.25)

        img_train1, img_val1, label_train1, label_val1 = train_val_split(X_train1, y_train1)

        X_train2, X_test2, y_train2, y_test2 = train_test_split(images, encoded_labels, test_size=0.25)

        img_train2, img_val2, label_train2, label_val2 = train_val_split(X_train2, y_train2)

        X_test1, X_test2, y_test1, y_test2 = np.array(X_test1), np.array(X_test2), np.array(y_test1), np.array(y_test2)

        for batch_size in BATCH_SIZES:
            for dense_units in DENSE_UNITS:
                for epochs in EPOCHS:
                    result = dict()
                    result["input size"] = input_size
                    result["batch size"] = batch_size
                    result["epochs"] = epochs
                    result["dense units"] = dense_units

                    model = create_VGG_classifier(input_size, dense_units)

                    history1 = History()
                    history1 = model.fit(img_train1, label_train1, batch_size=batch_size,
                                         validation_data=[img_val1, label_val1], callbacks=[history1], epochs=epochs)

                    test_loss1, test_acc1 = model.evaluate(X_test1, y_test1)

                    result["train loss 1"] = history1.history['loss']
                    result["train acc 1"] = history1.history['accuracy']
                    result["val loss 1"] = history1.history['val_loss']
                    result["val acc 1"] = history1.history['val_accuracy']
                    result["test loss 1"] = test_loss1
                    result["test acc 1"] = test_acc1

                    loss_acc_graphs(history1, f"split1_{input_size}_{batch_size}_{epochs}_{dense_units}")

                    model = create_VGG_classifier(input_size, dense_units)

                    history2 = History()
                    history2 = model.fit(img_train2, label_train2, batch_size=batch_size,
                                         validation_data=[img_val2, label_val2], callbacks=[history2], epochs=epochs)

                    test_loss2, test_acc2 = model.evaluate(X_test2, y_test2)

                    result["train loss 2"] = history2.history['loss']
                    result["train acc 2"] = history2.history['accuracy']
                    result["val loss 2"] = history2.history['val_loss']
                    result["val acc 2"] = history2.history['val_accuracy']
                    result["test loss 2"] = test_loss2
                    result["test acc 2"] = test_acc2

                    loss_acc_graphs(history2, f"split2_{input_size}_{batch_size}_{epochs}_{dense_units}")

                    loss_acc_graphs(history1, f"avg_{input_size}_{batch_size}_{epochs}_{dense_units}", log2=history2)

                    result_df = pd.DataFrame.from_dict(columns=results.columns, data={'0': result}, orient='index')
                    results = pd.concat([results, result_df], axis=0, ignore_index=True)

    results.to_csv('./results')


def create_labeled_data():
    img_id = 1
    for l in mat['labels'][0]:
        num_zeroes = math.ceil(4 - math.log10(img_id))
        path = './jpg/image_' + '0' * num_zeroes + '{}.jpg'.format(img_id)

        img = cv2.imread(path)

        dir_p = './labeled/label_' + str(l)
        filename = dir_p + '/image_' + '0' * num_zeroes + '{}.jpg'.format(img_id)

        if not os.path.exists(dir_p):
            Path(dir_p).mkdir(exist_ok=True, parents=True)

        cv2.imwrite(filename, img)

        img_id += 1


"""imgs = []
size = 640
image_path = "./jpg/image_00001.jpg"
image = Image.open(image_path).resize((size, size))
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)
imgs.append(image_tensor)

image_path = "./jpg/image_00002.jpg"
image = Image.open(image_path).resize((size, size))
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)
imgs.append(image_tensor)"""


size = 320
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()
])


image_paths = ["./jpg/image_00002.jpg",  "./jpg/image_00001.jpg"]

image_tensors = []
for image_path in image_paths:
    image = Image.open(image_path)
    image_tensor = transform(image)
    print(image_tensor.shape)
    image_tensors.append(image_tensor)

# Convert the list of image tensors into a single tensor
batch_size = len(image_tensors)
image_tensor = torch.stack(image_tensors, dim=0)
print(type(image_tensor) ,image_tensor.shape)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False, classes=102)
model.model = model.model[:10]

with torch.no_grad():
    output = model(image_tensor)
    print(output.numpy().shape)
