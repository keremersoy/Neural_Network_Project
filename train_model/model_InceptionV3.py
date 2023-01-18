import os
import os.path
p=str(os.path.dirname(os.path.abspath(__file__)))

from prepare_dataset import get_dataset

from keras.callbacks import EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np;

def train():
    global y_test, training_set, validation_set, test_set
    global CNN_inc,CNN_inc_history
    y_test, training_set, validation_set, test_set = get_dataset()

    cb = [EarlyStopping(monitor = 'loss', mode = 'min', patience = 5, restore_best_weights = True)]

    #InceptionV3 modeli oluşturma
    CNN_base_inc = InceptionV3(input_shape = (75, 75, 3), include_top = False, weights = 'imagenet')
    for layer in CNN_base_inc.layers:
        layer.trainable = False
    
    #Flattening
    x = layers.Flatten()(CNN_base_inc.output)
    
    #sinir ağı katmanları
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    CNN_inc = Model(CNN_base_inc.input, x)

    # Model Derleme
    CNN_inc.compile(optimizer = RMSprop(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Train
    CNN_inc_history = CNN_inc.fit(training_set, epochs = 10, validation_data = validation_set, callbacks = cb)


def test():
    acc = CNN_inc_history.history['accuracy']
    val_acc = CNN_inc_history.history['val_accuracy']
    loss = CNN_inc_history.history['loss']
    val_loss = CNN_inc_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.clf()
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    path='/temp/InceptionV3/acc_table'
    plt.savefig(p+path)

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    path='/temp/InceptionV3/loss_table'
    plt.savefig(p+path)

    score_inc = CNN_inc.evaluate(test_set)
    test_loss=score_inc[0]
    test_acc=score_inc[1]

    y_pred_inc = CNN_inc.predict(test_set)
    y_pred_inc = np.round(y_pred_inc)

    plt.clf()
    plt.figure(figsize = (6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_inc),annot = True, fmt = 'd')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path='/temp/InceptionV3/cm'
    plt.savefig(p+path)

    return test_acc, test_loss

def save(path):
    CNN_inc.save(path)


train()
test()
save(p+"./temp/InceptionV3.h5")


    