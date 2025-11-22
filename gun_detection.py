import numpy as np
import cv2 as cv
import imutils as iu
import datetime as dt

# Load cascade
gun_cascade = cv.CascadeClassifier("cascade.xml")

if gun_cascade.empty():
    print("Error: Cascade file not loaded!")
    exit()

camera = cv.VideoCapture(0)

gun_exist = False  # Default value

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = iu.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect guns
    gun = gun_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # If detection found
    if len(gun) > 0:
        gun_exist = True

    # Draw bounding boxes
    for (x, y, w, h) in gun:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow("Security Feed", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

# Final output
if gun_exist:
    print("Guns detected")
else:
    print("Guns not detected")
