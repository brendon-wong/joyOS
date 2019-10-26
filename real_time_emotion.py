from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import atexit


# Display emotion log after program exists
def exitProgram():
    # print average of list from each emotion in database
    #print(userEmotion)
    for key in userEmotion:
        userEmotion[key] = sum(userEmotion[key]) / len(userEmotion[key])
    #print(userEmotion)
    print("Emotion Percentage")
    print("---------------------------------")
    print("Anger: " + "{:.2f}%".format(userEmotion["angry"] * 100))
    print("Disgust: " + "{:.2f}%".format(userEmotion["disgust"] * 100))
    print("Scared: " + "{:.2f}%".format(userEmotion["scared"] * 100))
    print("Happy: " + "{:.2f}%".format(userEmotion["happy"] * 100))
    print("Sad: " + "{:.2f}%".format(userEmotion["sad"] * 100))
    print("Surprised: " + "{:.2f}%".format(userEmotion["surprised"] * 100))
    print("Neutral: " + "{:.2f}%".format(userEmotion["neutral"] * 100))

# parameters for loading data and images
detection_model_path = 'emotion_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion_models/_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# User Emotion Database
userEmotion = {
    'angry' : [],
    'disgust' : [],
    'scared' : [],
    'happy' : [],
    'sad' : [],
    'surprised' : [],
    'neutral' : []
}

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    preds = []
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                userEmotion[emotion].append(prob)
                # construct the label text               
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
atexit.register(exitProgram)