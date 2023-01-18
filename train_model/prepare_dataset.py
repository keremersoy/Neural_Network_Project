import os
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
def labelingDatas():
    p=str(os.path.dirname(os.path.abspath(__file__)))
    sdir=p+"\\..\\data"

    imagePath=[]
    labels=[]
    classlist=os.listdir(sdir)
    for klass in classlist:
        classpath=os.path.join(sdir,klass)
        if os.path.isdir(classpath):
            flist=os.listdir(classpath)
            for f in flist:
                fpath=os.path.join(classpath,f)
                imagePath.append(fpath)
                labels.append(klass)                   
    Fseries= pd.Series(imagePath, name='path')
    Lseries=pd.Series(labels, name='label')    
    dataset=pd.concat([Fseries, Lseries], axis=1)
    shuffeledData = dataset.sample(frac=1).reset_index()
    return shuffeledData

def imageDataGenerator(testset_df,trainset_df):
    LE = LabelEncoder()

    y_test = LE.fit_transform(testset_df["label"])

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.1,
                                        rotation_range = 20,
                                        width_shift_range = 0.1,
                                        height_shift_range = 0.1,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        validation_split = 0.1)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_dataframe(
    dataframe = trainset_df,
    x_col = "path",
    y_col = "label",
    target_size = (75, 75),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = 32,
    shuffle = True,
    seed = 2,
    subset = "training")

    validation_set = train_datagen.flow_from_dataframe(
    dataframe = trainset_df,
    x_col = "path",
    y_col = "label",
    target_size = (75, 75),
    color_mode ="rgb",
    class_mode = "binary",
    batch_size = 32,
    shuffle = True,
    seed = 2,
    subset = "validation")

    test_set = test_datagen.flow_from_dataframe(
    dataframe = testset_df,
    x_col = "path",
    y_col = "label",
    target_size = (75, 75),
    color_mode ="rgb",
    class_mode = "binary",
    shuffle = False,
    batch_size = 32)

    return y_test,training_set,validation_set,test_set

def get_dataset():
    data=labelingDatas()
    trainset_df, testset_df = train_test_split(data, train_size = 0.90, random_state = 42)
    return imageDataGenerator(testset_df,trainset_df)
