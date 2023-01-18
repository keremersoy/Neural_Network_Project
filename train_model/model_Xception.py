import os
import os.path
p=str(os.path.dirname(os.path.abspath(__file__)))

from prepare_dataset import get_dataset

from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Dense,GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.applications.xception import Xception

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np;

def train():
    global y_test, training_set, validation_set, test_set
    global CNN_xcep_history, CNN_xcep
    y_test, training_set, validation_set, test_set = get_dataset()
    print(type(y_test))
    cb = [EarlyStopping(monitor = 'loss', mode = 'min', patience = 5, restore_best_weights = True)]

    #Xception modeli oluşturma
    CNN_base_xcep = Xception(input_shape = (75, 75, 3), include_top = False, weights = 'imagenet')
    CNN_base_xcep.trainable = False
    
    #Sinir ağı katmanları
    CNN_xcep = Sequential()
    CNN_xcep.add(CNN_base_xcep)
    CNN_xcep.add(GlobalAveragePooling2D())
    CNN_xcep.add(Dense(128))
    CNN_xcep.add(Dropout(0.1))
    CNN_xcep.add(Dense(1, activation = 'sigmoid'))
    CNN_xcep.summary()

    # derleme
    CNN_xcep.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])

    # train
    CNN_xcep_history = CNN_xcep.fit(training_set, epochs = 10, validation_data = validation_set, callbacks = cb)


def test():
    acc = CNN_xcep_history.history['accuracy']
    val_acc = CNN_xcep_history.history['val_accuracy']
    loss = CNN_xcep_history.history['loss']
    val_loss = CNN_xcep_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.clf()
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    path='/temp/Xception/acc_table'
    plt.savefig(p+path)

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    path='/temp/Xception/loss_table'
    plt.savefig(p+path)

    score_xcep = CNN_xcep.evaluate(test_set)
    test_loss=score_xcep[0]
    test_acc=score_xcep[1]

    y_pred_xcep = CNN_xcep.predict(test_set)
    y_pred_xcep = np.round(y_pred_xcep)
    
    plt.clf()
    plt.figure(figsize = (6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_xcep),annot = True, fmt = 'd')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path='/temp/Xception/cm'
    plt.savefig(p+path)

    return test_acc, test_loss

def save(path):
    CNN_xcep.save(path)


train()
test()
save(p+"./temp/Xception.h5")