import numpy as np
import cv2
from time import sleep
import tensorflow as tf

np.set_printoptions(suppress=True)

labels = ['no_mask','mask','background']
model = tf.keras.models.load_model('maskmodel.h5')

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if success == True:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = cv2.flip(image,1) #左右反轉
        img = cv2.resize(image,(224,224))
        img = np.array(img,dtype=np.float32)
        img = np.expand_dims(img,axis=0)
        img = (img/127.0) - 1 #正規化
        data[0] = img

        prediction = model.predict(data) #預測
        print(prediction)
        predicted_class = np.argmax(prediction[0], axis=-1)
        predicted_class_name = labels[predicted_class]
        print(predicted_class_name)
		
        if predicted_class_name == 'no_mask':
            cv2.putText(image, 'Please wear a mask', (10, 50), cv2.FONT_HERSHEY_DUPLEX,
            1, (0, 0, 255), 1, cv2.LINE_AA)
            
        elif predicted_class_name == 'mask':    
            cv2.putText(image, 'You can pass.', (10, 50), cv2.FONT_HERSHEY_DUPLEX,
            1, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Frame",image)
        sleep(1)
		           	
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()	    	