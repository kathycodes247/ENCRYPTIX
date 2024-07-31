# Face Detection and Recognition

This project develops an AI application that can detect and recognize faces in images or videos. It uses pre-trained face detection models like Haar cascades or deep learning-based face detectors. Optionally, it adds face recognition capabilities using techniques like Siamese networks or ArcFace.

**Installation**

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/face-detection-recognition.git
cd face-detection-recognition
pip install -r requirements.txt
```
**Usage**
_Face Detection_

Face detection can be performed using Haar cascades or deep learning-based detectors like MTCNN. Here is an example using OpenCV's Haar cascades.
```python
import cv2

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_faces('path_to_image.jpg')
```
_Face Recognition_

Face recognition can be implemented using techniques like Siamese networks or ArcFace. Here is an example using the face_recognition library.

```python
import face_recognition

# Load the known images
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load the unknown image
unknown_image = face_recognition.load_image_file("unknown_person.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces([known_encoding], unknown_encoding)

if results[0]:
    print("It's a match!")
else:
    print("No match found.")
```

_Detect and Recognize Faces in Video_
Here's how you can detect and recognize faces in a video stream:
```python
import cv2
import face_recognition

# Load the known images and encode faces
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_faces = [known_encoding]

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Known Person"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
```

**Contributing**
If you'd like to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature-name).
3. Make your changes and commit them (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature-name).
5. Open a pull request.

**Contact**
For any questions or suggestions, feel free to reach out:

@GitHub: kathycodes247

