#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


pip install keras


# In[3]:


from keras.preprocessing import image


# In[4]:


categories=['with_mask','without_mask']


# In[5]:


import os


# In[6]:


pip install opencv


# In[7]:


#OpenCV is a Python library that allows you to perform image processing and computer vision tasks
import cv2


# In[8]:


path='C:\\Users\\ASUS\\Downloads\\Github_masknomask\\train'


# In[9]:


data=[]

for category in categories:


  #joins path +category to create new path 
  new_path=os.path.join(path,category)

#will print the index of the category of img
  label=categories.index(category)

#os.listdir(path) returns the list of all the files in that path(since we cannot directly add folder to colab)
  for file in os.listdir(new_path):
    
    img_path=os.path.join(new_path,file)
    #imread loads img
    img=cv2.imread(img_path)
    #since vgg16 takes this size
    img=cv2.resize(img,(224,224))

    data.append([img,label])

    
    #here every image is rgb image
    #we need two columns here one with numpy array of img and other with category of img (0-with mask,1-without mask)

    # we need 2-d numpy array for img and 1d array for label


# In[10]:


import random


# In[11]:


#to shuffle images
random.shuffle(data)


# In[12]:


#The X and y get converted to array hence for next execution the variable has to be modified



# In[13]:


#now we need to separate x and y and convert them into numpy array

X=[]
y=[]

 
for features,label in data:
  X.append(features)
  y.append(label)


# In[14]:


import numpy as np


# In[15]:


#Here we have made X and y as array from list

X=np.array(X) 
y=np.array(y)


# In[16]:


X.shape


# In[17]:


#standardize X

X=X/255


# In[18]:


#Splitting train test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


#We will perform transfer learing of vgg16


# In[22]:


from keras.applications.vgg16 import VGG16


# In[23]:


vgg=VGG16()


# In[24]:


vgg.summary()


# In[25]:


#We will replace the last dense block and use the model again

#VGG16 is an functional model(any layer can be connected to any other layer) hence we used this sequential method to have sequential (one layer to next layer only)


# In[26]:


from keras import Sequential


# In[27]:


#using simple sequential keras
model=Sequential()


# In[28]:


#for adding layers other than the last dense layer

for layer in vgg.layers[:-1]:
  model.add(layer)


# In[29]:


model.summary()


# In[30]:


#Now we will freeze the weights of the layers so that they don't get updated during training


# In[31]:


for layer in model.layers:
  layer.trainable=False


# In[32]:


#trainable parameters =0 now


# In[33]:


model.summary()


# In[34]:


#Add dense layer now

from keras.layers import Dense
#1 is no of output node
model.add(Dense(1,activation='sigmoid'))


# In[35]:


#trainable parameters and last layer updated


model.summary()


# In[36]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[37]:


X_test.shape


# In[38]:


model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))


# In[39]:





# In[76]:


def detect_face_mask(img):
    
    
    #y_pred=model.predict_classes(img.reshape(1,224,224,3))
    
    input_image_reshaped = np.reshape(img, (1,224,224,3))
     
    input_prediction = model.predict(input_image_reshaped)
    
    print (input_prediction[0][0])
    
    
    if (input_prediction[0][0]<0.5):
             y_pred=0
    else:
        y_pred=1
    print(y_pred)  


# In[83]:


#used for testing if model is working fine and the def of detect_face_mask is same as mentioned in spyder
sample1=cv2.imread('C:\\Users\\ASUS\\Downloads\\with mask\\download.jpg')
sample1=cv2.resize(sample1,(224,224))


# In[84]:


detect_face_mask(sample1)


# In[65]:


#use this instead of pickle to save in the desired folder
model.save('C:\\Users\\ASUS\\Downloads\\keras')


# In[ ]:




