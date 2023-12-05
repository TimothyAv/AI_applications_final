import numpy as np
import time
import cv2
import os
from PIL import Image
from playsound import playsound
import threading
import smtplib
from email.message import EmailMessage

def alert():
    threading.Thread(target=playsound, args=('/Users/nika/Downloads/alarmm.wav',), daemon=True).start()

def send_email(frame):
    Sender_Email = "nigaram8@gmail.com"
    Reciever_Email = "nigarannn@gmail.com"
    Password = 'juup czbr jviq mfaw'  # ENTER GOOGLE APP PASSWORD HERE

    newMessage = EmailMessage()
    newMessage['Subject'] = "Animal Detected"
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email
    newMessage.set_content('An animal has been detected')

    # Save the frame as an image
    _, image_data = cv2.imencode('.png', frame)
    image_type = 'png'
    image_name = 'detected_animal.png'

    newMessage.add_attachment(image_data.tobytes(), maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)
        smtp.send_message(newMessage)

def async_email(frame):
    threading.Thread(target=send_email, args=(frame.copy(),), daemon=True).start()

def reset_alarm_flag():
    global alarm_flag
    alarm_flag = False

def reset_alarm_cooldown():
    global alarm_cooldown
    alarm_cooldown = False

args = {"confidence": 0.5, "threshold": 0.3}
alarm_flag = False  # Initialize alarm flag to False
alarm_cooldown = False  # Initialize alarm cooldown to False
detected_objects = set()

labelsPath = "/Users/nika/Downloads/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['bird', 'cat', 'dog', 'sheep', 'horse', 'cow', 'elephant', 'zebra', 'bear', 'giraffe']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.abspath("/Users/nika/Downloads/yolov3-tiny.weights")
configPath = os.path.abspath("/Users/nika/Downloads/yolov3-tiny.cfg")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getUnconnectedOutLayersNames()

vs = cv2.VideoCapture("/Users/nika/Downloads/bird.mp4")
#vs = cv2.VideoCapture("/Users/nika/Downloads/doggg.mp4")
#vs = cv2.VideoCapture(0)

alarm_timer = None
cooldown_timer = None

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (x, y, w, h) = box.astype("int")

                # Calculate the top-left corner (x, y) and the bottom-right corner (x + w, y + h) of the bounding box
                x = int(x - w / 2)
                y = int(y - h / 2)
                w = int(w)
                h = int(h)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            if LABELS[classIDs[i]] in final_classes:
                (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

                if (x, y, w, h) not in detected_objects:
                    if not alarm_flag and not alarm_cooldown:
                        alert()
                        async_email(frame)
                        alarm_flag = True
                        alarm_cooldown = True
                        # Set a timer to reset alarm_flag after 10 seconds (adjust as needed)
                        alarm_timer = threading.Timer(10.0, reset_alarm_flag)
                        alarm_timer.start()
                        # Set a timer to reset alarm_cooldown after 60 seconds (adjust as needed)
                        cooldown_timer = threading.Timer(60.0, reset_alarm_cooldown)
                        cooldown_timer.start()

                    detected_objects.add((x, y, w, h))

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # Reset alarm flag and cooldown when no objects are detected
        alarm_flag = False
        detected_objects.clear()

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
