# ⭐ README.md — Gun Detection System using Haar Cascade & OpenCV
## 📌 Project Overview

This project is a real-time Gun Detection System built using:

- Python

- OpenCV

- Haar Cascade Classifier

- Imutils

The system uses a pre-trained Haar Cascade (cascade.xml) to detect firearms from a webcam feed.
When a gun-like object is detected, the system draws bounding boxes and reports detection.

This project is ideal for Computer Vision beginners, OpenCV learners, and portfolio building.

## 🎯 Features

✔ Real-time gun detection using webcam
✔ Haar Cascade classifier for fast detection
✔ Live bounding boxes around detected objects
✔ Custom cascade support
✔ Lightweight & efficient
✔ Beginner-friendly code with comments

## 🧠 How Haar Cascade Works (Short Understanding)
Haar Cascade works in 3 steps:

1️⃣ Haar Features (Light–Dark Rectangle Patterns)

It checks brightness differences in rectangles to detect shapes (edges, lines, curves).

2️⃣ Sliding Window

A 40×40 window scans the image from left → right → down at multiple scales.

3️⃣ Cascade Stages (Checkpoints)

Each patch must pass 10+ stages:

Stage 1 → simple edge check

Stage 2 → more detailed features

…

Final Stage → confirm object

If a window passes all stages → gun detected ✔

## 🖥️ Tech Stack
| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Computer Vision      | OpenCV                       |
| Model                | Haar Cascade (cascade.xml)   |
| Helper Library       | Imutils                      |
| Platform             | Works on Windows, Mac, Linux |

## 📂 Project Structure
```text
📁 Gun-Detection-HaarCascade
│
├── gun_detection.py        # Main Python script
├── cascade.xml             # Haar cascade model for gun detection
├── README.md               # Documentation
└── sample_output.png       # Screenshot (optional)
```


## 🚀 How to Run the Project
1️⃣ Install Dependencies
'''text
pip install opencv-python imutils numpy
'''

2️⃣ Keep the files together
gun_detection.py
cascade.xml

3️⃣ Run the script
python gun_detection.py

4️⃣ Quit the video

Press Q to exit.

🧩 Complete Python Code
import numpy as np
import cv2 as cv
import imutils as iu
import datetime as dt

gun_cascade = cv.CascadeClassifier("cascade.xml")
camera = cv.VideoCapture(0)

gun_exist = False

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = iu.resize(frame, width=500)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gun = gun_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(gun) > 0:
        gun_exist = True

    for (x, y, w, h) in gun:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow("Security Feed", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

if gun_exist:
    print("Guns detected")
else:
    print("No guns detected")

📊 Concept Understanding (Step-by-Step)
🔹 Step 1 — Load Haar Cascade

This file is the "brain" that contains trained object features.

🔹 Step 2 — Access Webcam

OpenCV captures live video frames.

🔹 Step 3 — Convert Frame to Grayscale

Required for Haar features (they work on intensity only).

🔹 Step 4 — Apply detectMultiScale()

This function:

Slides the window across frame

Checks light–dark patterns

Runs through cascade stages

Marks detection

🔹 Step 5 — Draw Detection Box

A rectangle is drawn where the gun is found.

🔹 Step 6 — Display Output

Shows live security feed with bounding boxes.

🔹 Step 7 — Final Output

Prints whether any gun was detected during your session.

📈 Diagrams & Explanation
1️⃣ Haar Features Diagram
+------+------+
| DARK | LIGHT |
+------+------+
Edge detection

2️⃣ Sliding Window Scan
Row 1: [WIN] → → → →
Row 2: ↓ [WIN] → → →
Row 3: ↓ [WIN] → → →

3️⃣ Cascade Stages
Stage 1 → Stage 2 → … → Stage N
(Passes all?) → Gun Detected ✔

🛡️ Limitations

❌ Haar Cascades are not fully accurate
❌ Works best in good lighting
❌ Should not be used for real security without ML upgrades

🔮 Future Improvements

✔ Switch to YOLOv8 / YOLOv9 gun detection (very accurate)
✔ Add alarm system on detection
✔ Add image recording + timestamp
✔ Add email/mobile alert system

📜 License

This project is free to use under the MIT License.

❤️ Author

SK SAMIM AKHTAR
Python Learner | Data Science Learner | Computer Vision Enthusiast

If you want:

📘 Convert this README into PDF
🎨 Add images or badges (GitHub shields)
🚀 Make this an advanced computer vision portfolio project
🟩 Improve accuracy using YOLO

Just tell me — I will make it!
