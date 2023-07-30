# https://youtu.be/XIrOM9oP3pA is the source

import cv2

#Our Image
imgFile = 'carTraffic.jpeg'

# pre-trained car classifier (Source: https://github.com/Kalebu/Real-time-Vehicle-Dection-Python/blob/master/haarcascade_car.xml?ref=hackernoon.com)
classifierFile = 'haarcascade_car.xml'

#create opencv image
img = cv2.imread(imgFile)

#create car classifier
carTracker = cv2.CascadeClassifier(classifierFile)

#turn image into grayscale (for haar cascade)
blacknWhite = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = carTracker.detectMultiScale(blacknWhite)
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#draw rectangle around cars

#display image with cars spotted
cv2.imshow("Clever Programmer Car Detector", img)

#don't autoclose, wait for keypress
cv2.waitKey()

print("Code Completed")