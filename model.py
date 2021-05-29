from tensorflow.keras.layers import Conv2D,MaxPool2D,GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.metrics import Precision, Recall
import datetime
import time



t_set = lambda: datetime.datetime.now().astimezone()
t_diff = lambda t: str(t_set() - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

logger_t.info(f'{"="*100} \n\nSTEP 5: Create Model and train the model\n')

def create_model():
    model = tf.keras.Sequential([
                  tf.keras.Input(shape=(224,224,1)),
                  Conv2D(32,3,strides=2,padding='same',activation='relu',use_bias=False),
                  MaxPool2D(),
                  Conv2D(64,3,strides=2,padding='same',activation='relu',use_bias=False),
                  MaxPool2D(),
                  GlobalAveragePooling2D(),
                  Dense(128),
                  Dropout(0.5),
                  Dense(64),
                  Dropout(0.5),
                  Dense(2,activation='sigmoid')])
    return model