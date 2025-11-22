# python_gun_detection
# ⭐ Gun Detection System using Haar Cascade & OpenCV

A real-time Gun Detection System built using Python, OpenCV, and a Haar Cascade model.
The system detects firearms from a live webcam feed, draws bounding boxes, and prints detection results.

Perfect project for Computer Vision learning, OpenCV practice, and GitHub portfolio building.

📌 Features

🔍 Real-time gun detection

⚡ Fast Haar Cascade classifier

🎥 Uses webcam live feed

📦 Lightweight, no deep learning required

🧩 Beginner-friendly and well-structured

🧠 How Haar Cascade Works (Simple Explanation)
1️⃣ Haar Features

Haar features compare light vs dark rectangular regions.

Example:

+-------+-------+
| DARK  | LIGHT |
+-------+-------+

2️⃣ Sliding Window

A 40×40 window moves across the frame:

Row 1: [WIN] → → → →
Row 2: ↓ [WIN] → → → →
Row 3: ↓ [WIN] → → → →

3️⃣ Cascade Stages (Checkpoints)

Every patch goes through multiple “stages”:

Stage 1 → Stage 2 → Stage 3 → ... → Final Stage
If passed → Gun DETECTED ✔

🖥️ Tech Stack
Component	Technology
Language	Python
Vision Library	OpenCV
Model	Haar Cascade (cascade.xml)
Helper Tool	Imutils
Platform	Windows / Mac / Linux
📂 Project Structure
📁 Gun-Detection-HaarCascade
│
├── gun_detection.py                        # Main detection script
├── cascade.xml                             # Haar cascade model
├── requirements.txt                         # Required packages
├── README.md                                # Documentation
│
├── docs/                                    # Project explanation PDFs
│   ├── haar_cascade_explanation.pdf
│   ├── haar_cascade_styled.pdf
│   └── haar_full_explanation_advanced.pdf
│
├── assets/                                  # Images for documentation
│   └── sample_output.png
│
├── .gitignore                               # Ignore unnecessary files
└── LICENSE                                   # MIT License

🚀 How to Run the Project
1️⃣ Install Required Packages
pip install -r requirements.txt


Or manually:

pip install opencv-python numpy imutils

2️⃣ Make Sure These Files Are Together
gun_detection.py
cascade.xml

3️⃣ Run the Project
python gun_detection.py

4️⃣ Exit

Press Q to close the webcam window.

🧩 Complete Code (Copy & Paste)
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

📊 Understanding the Project (Step-by-Step)
🔹 Step 1 — Load Haar Cascade

The classifier (cascade.xml) contains pre-trained patterns of the gun.

🔹 Step 2 — Start Webcam Feed

Frames are captured in real-time.

🔹 Step 3 — Convert to Grayscale

Required for Haar feature comparison.

🔹 Step 4 — detectMultiScale()

Runs:

sliding window

Haar feature checks

cascade stages

Returns (x, y, w, h) if object detected.

🔹 Step 5 — Draw Rectangle

Bounding box is placed on the detected object.

🔹 Step 6 — Display Feed

Live window shows detections.

👁️ Example Output

(Add image in assets/sample_output.png)

┌─────────────────────────────────────┐
│   [   GUN DETECTED BOUNDING BOX ]   │
└─────────────────────────────────────┘

🔮 Future Improvements

Upgrade to YOLOv8/YOLOv9 for higher accuracy

Add buzzer alarm on detection

Save detected frames with timestamp

Email/SMS alerts for security use

📜 License

This project is licensed under the MIT License.

MIT License  
Copyright (c) 2025  
SK SAMIM AKHTAR

✨ Author

SK NAIM AKHTAR
Python Developer • Data Scientist (Learning) • Computer Vision Enthusiast
