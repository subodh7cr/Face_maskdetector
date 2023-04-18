import numpy as np

import cv2

def detect_face_mask(img):
    
    #y_pred=model.predict_classes(img.reshape(1,224,224,3))
    
    input_image_reshaped = np.reshape(img, (1,224,224,3))
     
    input_prediction = model.predict(input_image_reshaped)
    
    print (input_prediction[0][0])
    
    #we used this condition of less than because when we use argmax code(replacement of predict_classes) in that case 
    #all the values have max as 0 so 0 will always be the output
    
    
    if (input_prediction[0][0]<0.0001):
             y_pred=0
    else:
        y_pred=1
    print(y_pred)  
        
        
    return y_pred 
  
  
  def draw_label(img,text,pos,bg_color):
     
     
     
    text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)

    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    
  haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
    
  def detect_face(img):
    
    coods= haar.detectMultiScale(img)
    
    return coods  
  
  
  from tensorflow import keras

#to import the saved keras model
model=keras.models.load_model("C:\\Users\\ASUS\\Downloads\\keras")

cap = cv2.VideoCapture(0)
while True:
    
    
    #ret -> T/F value , and img in frame
    ret,frame=cap.read()

    # we have to call the detection method

    # here changing from frame to img will let the webcam open in large size
    img=cv2.resize(frame,(224,224))
    
    img_true=detect_face_mask(img)
    
    #for face detection , hence has to be converted to grey img
    coods= detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    
    for x,y,w,h in coods:
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    
    
    if img_true==0:
        
        #for label of mask no mask
        draw_label(frame,"mask",(30,30),(0,255,0))
    else:
        draw_label(frame,"No mask",(30,30),(0,0,255))
        
     
    
