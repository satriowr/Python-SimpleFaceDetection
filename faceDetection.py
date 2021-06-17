import cv2

faceCascade = cv2.CascadeClassifier('SimpleFaceDetection\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0) #webcam

while True :
    _, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in face :
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('faceDetection', img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
