import tensorflow as tf
import keras
import os



import glob
import matplotlib.pyplot as plt
import numpy as np
# Keras API
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


test_dir ="test_set"
train_dir="train_set"


def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count


train_samples =get_files(train_dir)
num_classes=len(glob.glob(train_dir+"/*"))
test_samples=get_files(test_dir)
#print(num_classes,"Classes")
#print(train_samples,"Train images")
#print(test_samples,"Test images")


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)  


# set height and width and color of input image.
img_width,img_height =256,256
input_shape=(img_width,img_height,3)
batch_size =32
train_generator =train_datagen.flow_from_directory(train_dir,target_size=(img_width,img_height),batch_size=batch_size)
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,target_size=(img_width,img_height),batch_size=batch_size)

#train_generator.class_indices


model = Sequential()
model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))          
model.add(Dense(num_classes,activation='softmax'))
#model.summary()


from keras.preprocessing import image
import numpy as np
img1 = image.load_img('train_set/Potato___Early_blight/2.JPG')
#plt.imshow(img1);
#preprocess image
img1 = image.load_img('Potato___Early_blight/2.JPG', target_size=(256, 256))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)


from keras.models import Model
conv2d_output = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
max_pooling2d_output = Model(inputs=model.input,outputs=model.get_layer('max_pooling2d').output)
conv2d_1_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_1').output)
max_pooling2d_1_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_1').output)
conv2d_2_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_2').output)
max_pooling2d_2_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_2').output)
flatten_1_output=Model(inputs=model.input,outputs=model.get_layer('flatten').output)
conv2d_1_features = conv2d_output.predict(img)
max_pooling2d_1_features = max_pooling2d_output.predict(img)
conv2d_2_features = conv2d_1_output.predict(img)
max_pooling2d_2_features = max_pooling2d_1_output.predict(img)
conv2d_3_features = conv2d_2_output.predict(img)
max_pooling2d_3_features = max_pooling2d_2_output.predict(img)
flatten_1_features = flatten_1_output.predict(img)


validation_generator = train_datagen.flow_from_directory(train_dir,target_size=(img_height, img_width),batch_size=batch_size)
opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
train=model.fit(train_generator,steps_per_epoch=train_generator.samples//batch_size,validation_data=validation_generator)


# Save model
from keras.models import load_model
model.save('crop.h5')

# Loading model and predict.
from keras.models import load_model
model=load_model('crop.h5')
# Mention name of the disease into list.
Classes = ["Potato___Early_blight","Potato___Late_blight","Potato___healthy"]

import numpy as np
import matplotlib.pyplot as plt
# Pre-Processing test data same as train data.
img_width=256
img_height=256
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing import image
def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)
    
    
result = model.predict_classes([prepare('test_set/Potato___Early_blight/2.jpg')])
disease=image.load_img('test_set/Potato___Early_blight/2.jpg')
#plt.imshow(disease)
print (Classes[int(result)])