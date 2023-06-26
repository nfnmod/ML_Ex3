import os
import scipy.io
import numpy as np
import tensorflow
from keras import Input, Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import History
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.io as pio

IMAGES_PATH = "./jpg/"
PATH_GRAPHS = "./graphs"
RELU = "relu"
SOFTMAX = "softmax"
CAT_CROSS_EN = "categorical_crossentropy"
ACC = "accuracy"
TRAIN_VAL_SPLIT = 1 / 3

mat = scipy.io.loadmat('./imagelabels.mat')
print(mat)

VGG = VGG19(include_top=False, classes=102)
VGG.trainable = False
print(VGG.summary())

DENSE_UNITS = [32, 64, 128]
BATCH_SIZES = [32, 64, 128]
INPUT_SIZE = [224]


def preprocess(size):
    images = []
    for img_file in os.listdir(IMAGES_PATH):
        # load an image from file
        image = load_img(IMAGES_PATH + img_file, target_size=(size, size))

        # convert the image pixels to a numpy array
        image = img_to_array(image)

        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # prepare the image for the VGG model
        image = preprocess_input(image)

        images.append(image)

    return np.array(images)


def create_VGG_classifier(dense_units, lr=0.001):
    inp = Input()

    vgg_res = VGG(inp)

    dense = Dense(dense_units, activation='relu')(vgg_res)

    output = Dense(dense_units, activation='softmax')(dense)

    model = Model(inp, output)

    model.compile(optimizer=Adam(lr), loss=CAT_CROSS_EN, metrics=[ACC])

    return model


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


def loss_acc_graphs(log, hyper_paramters):
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
        y=log.history['loss'],
        mode='lines',
        name='train'
    ))
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(log.history['val_loss']) + 1)),
        y=log.history['val_loss'],
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
        y=log.history['accuracy'],
        mode='lines',
        name='train'
    ))
    fig_acc.add_trace(go.Scatter(
        x=list(range(1, len(log.history['val_accuracy']) + 1)),
        y=log.history['val_accuracy'],
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
    for input_size in INPUT_SIZE:
        images = preprocess(input_size)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25)

        img_train, img_val, label_train, label_val = train_val_split(X_train, y_train)

        print("train size {}, val size: {}, test size: {}")

        for batch_size in BATCH_SIZES:
            for dense_units in DENSE_UNITS:
                model = create_VGG_classifier(dense_units)
                history = History()
                history = model.fit(img_train, label_train, batch_size=batch_size, validation_data=[img_val, label_val], callbacks=[history])
                loss_acc_graphs(history, f"{input_size}_{batch_size}_{dense_units}")
