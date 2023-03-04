import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:

    check,frame = video.read()
    frame = cv2.flip(frame, 1)

    img = cv2.resize(frame,(224,224))

    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)

    normalised_image = test_image/255.0
    prediction = model.predict(normalised_image)
      
   # print(prediction)
    print("Rock : ",round(prediction[0,0]*100),"%",", Paper : ",round(prediction[0,1]*100),"%",", Scissors : ",round(prediction[0,2]*100),"%") 
        
    cv2.imshow("output",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Closing")
        break

video.release()