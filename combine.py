# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:59:57 2022

@author: User
"""


import audio_model
import image_model


def test_model(img_path):
  image_model.img = image_model.image.load_img(img_path,target_size=(100,100))
  
  x1=image_model.image.img_to_array(image_model.img)
  x1=x1/255
  x1=image_model.np.expand_dims(x1,axis=0)
  pred1 = image_model.np.argmax(image_model.model.predict(x1)[0], axis=-1)
  if pred1==0:
    print('Angry')
  elif pred1==1:
    print('Defense')
  elif pred1==2:
    print('Fighting')
  elif pred1==3:
    print('HuntingMind')
  elif pred1==4:
    print('Mating')
  elif pred1==5:
    print('Resting')
  else:
    print('Unknown')



img_path='C:/Users/User/Desktop/Latest-GP/Actual-Running/Combined-Project/f.jpg'

test_model(img_path)

