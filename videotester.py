import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("best_model.h5")


F_H_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


CAP = cv2.VideoCapture(0)

while True:
    ret, test_img = CAP.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = F_H_CASCADE.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        Roi_G = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        Roi_G = cv2.resize(Roi_G, (224, 224))
        pixels_image = image.img_to_array(Roi_G)
        pixels_image = np.expand_dims(pixels_image, axis=0)
        pixels_image /= 255

        P = model.predict(pixels_image)#prediction

        # find max indexed array
        max_index = np.argmax(P[0])

        Emotion = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = Emotion[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    image_resized = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', image_resized)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

CAP.release()
cv2.destroyAllWindows