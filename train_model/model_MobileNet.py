import os
import os.path
p=str(os.path.dirname(os.path.abspath(__file__)))

from prepare_dataset import get_dataset

from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Dense,GlobalAveragePooling2D,BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.applications.mobilenet import MobileNet

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np;
def train():
    global y_test, training_set, validation_set, test_set
    global CNN_mobilenet_history, CNN_mobilenet
    y_test, training_set, validation_set, test_set = get_dataset()
    print(type(y_test))
    cb = [EarlyStopping(monitor = 'loss', mode = 'min', patience = 5, restore_best_weights = True)]

    #MobileNet modeli oluşturma
    CNN_base_mobilenet = MobileNet(input_shape = (75, 75, 3), include_top = False, weights = 'imagenet')
    
    for layer in CNN_base_mobilenet.layers:
        layer.trainable = False
    
    #Sinir ağı katmanları
    CNN_mobilenet = Sequential()
    CNN_mobilenet.add(BatchNormalization(input_shape = (75, 75, 3)))
    CNN_mobilenet.add(CNN_base_mobilenet)
    CNN_mobilenet.add(BatchNormalization())
    CNN_mobilenet.add(GlobalAveragePooling2D())
    CNN_mobilenet.add(Dropout(0.5))
    CNN_mobilenet.add(Dense(1, activation = 'sigmoid'))    
    CNN_mobilenet.summary()
    
    # derleme
    CNN_mobilenet.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])
    
    # train
    CNN_mobilenet_history = CNN_mobilenet.fit(training_set, epochs = 2, validation_data = validation_set, callbacks = cb)

def test():
    acc = CNN_mobilenet_history.history['accuracy']
    val_acc = CNN_mobilenet_history.history['val_accuracy']
    loss = CNN_mobilenet_history.history['loss']
    val_loss = CNN_mobilenet_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.clf()
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    path='/temp/MobileNet/acc_table'
    plt.savefig(p+path)

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    path='/temp/MobileNet/loss_table'
    plt.savefig(p+path)

    score_mn = CNN_mobilenet.evaluate(test_set)
    test_loss=score_mn[0]
    test_acc=score_mn[1]

    y_pred_mn  = CNN_mobilenet.predict(test_set)
    y_pred_mn  = np.round(y_pred_mn )

    plt.clf()
    plt.figure(figsize = (6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_mn ),annot = True, fmt = 'd')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path='/temp/MobileNet/cm'
    plt.savefig(p+path)

    return test_acc, test_loss

def save(path):
    CNN_mobilenet.save(path)
train()
test()
save(p+"./temp/MobileNet.h5")
    