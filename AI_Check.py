import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
Model=keras.models.Sequential
Conv2D=keras.layers.Conv2D
MaxPooling2D=keras.layers.MaxPooling2D
Flatten=keras.layers.Flatten
Dense=keras.layers.Dense
Dropout=keras.layers.Dropout
Adam=keras.optimizers.Adam
base_dir=r"C:\Users\houdo\Desktop\New"
photo_dir=os.path.join(base_dir,'Photo')
img_width,img_height=150,150
batch_size=32
train_ds=tf.keras.utils.image_dataset_from_directory(base_dir,validation_split=0.2,subset="training",seed=123,image_size=(img_height,img_width),batch_size=batch_size,label_mode='binary')
val_ds=tf.keras.utils.image_dataset_from_directory(base_dir,validation_split=0.2,subset="validation",seed=123,image_size=(img_height,img_width),batch_size=batch_size,label_mode='binary')
data_augmentation=keras.Sequential([keras.layers.RandomFlip("horizontal"),keras.layers.RandomRotation(0.2),keras.layers.RandomZoom(0.2),keras.layers.RandomContrast(0.2)])
normalization_layer=keras.layers.Rescaling(1./255)
def process_image(image,label):
    image=data_augmentation(image,training=True)
    image=normalization_layer(image)
    return image,label
AUTOTUNE=tf.data.AUTOTUNE
train_ds=train_ds.map(process_image,num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds=val_ds.map(lambda x,y:(normalization_layer(x),y),num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
model=Model()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_width,img_height,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
epochs=10#太多了电脑真的跑不动
history=model.fit(train_ds,validation_data=val_ds,epochs=epochs)
model.save('ai_vs_photo_model.h5')
def predict_image(image_path,model):
    img=tf.keras.utils.load_img(image_path,target_size=(img_width,img_height))
    img_array=tf.keras.utils.img_to_array(img)
    img_array=tf.expand_dims(img_array,axis=0)
    img_array=normalization_layer(img_array)
    prediction=model.predict(img_array)
    return 'AI' if prediction[0][0] > 0.5 else 'Photo'
test_image_path =r"C:\Users\houdo\Desktop\AI_44.jpg"
loaded_model=keras.models.load_model('ai_vs_photo_model.h5')
result=predict_image(test_image_path,loaded_model)
print(f'The image is classified as:{result}')