import numpy as np
import cv2
import os
from PIL import Image
from playsound import playsound
import threading
import smtplib
from email.message import EmailMessage

haar_cascade = r'C:\Users\rrakh\Downloads\Telegram Desktop\cars.xml'
car_cascade = cv2.CascadeClassifier(haar_cascade)

def alert():
    threading.Thread(target=playsound, args=(r"C:\Users\rrakh\Downloads\Telegram Desktop\alarmm.wav",), daemon=True).start()

def send_email(frame):
    Sender_Email = "nigaram8@gmail.com"
    Reciever_Email = "nigarannn@gmail.com"
    Password = 'juup czbr jviq mfaw'  #ENTER GOOGLE APP PASSWORD HERE.

    newMessage = EmailMessage()
    newMessage['Subject'] = "Object Detected"
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email
    newMessage.set_content('An object has been detected')

    #Save the frame as an image.
    _, image_data = cv2.imencode('.png', frame)
    image_type = 'png'
    image_name = 'detected_object.png'

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
alarm_flag = False  #Initialize alarm flag to False.
alarm_cooldown = False  #Initialize alarm cooldown to False.
detected_objects = set()

labelsPath = r"C:\Users\rrakh\Downloads\Telegram Desktop\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['bird', 'cat', 'dog', 'sheep', 'horse', 'cow', 'elephant', 'zebra', 'bear', 'giraffe']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath_yolo = os.path.abspath(r"C:\Users\rrakh\Downloads\Telegram Desktop\yolov3-tiny.weights")
configPath_yolo = os.path.abspath(r"C:\Users\rrakh\Downloads\Telegram Desktop\yolov3-tiny.cfg")
net_yolo = cv2.dnn.readNetFromDarknet(configPath_yolo, weightsPath_yolo)
layer_names_yolo = net_yolo.getUnconnectedOutLayersNames()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

#Open the video.
cap = cv2.VideoCapture(r'C:\Users\rrakh\Desktop\1.mp4')

out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

while True:
    #Capture frames.
    ret, frame = cap.read()

    if not ret:
        break

    (H, W) = frame.shape[:2]

    #Resize the frame if needed.
    resize_width = 640
    frame = cv2.resize(frame, (resize_width, int(resize_width * H / W)))

    #Convert frames to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect cars of different sizes in the input image.
    cars = car_cascade.detectMultiScale(gray, 1.1, 10)

    #Draw rectangles for cars.
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #YOLO object detection.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net_yolo.setInput(blob)
    layerOutputs = net_yolo.forward(layer_names_yolo)

    #Process YOLO detections.
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
                        #Set a timer to reset alarm_flag after 10 seconds.
                        alarm_timer = threading.Timer(10.0, reset_alarm_flag)
                        alarm_timer.start()
                        #Set a timer to reset alarm_cooldown after 60 seconds.
                        cooldown_timer = threading.Timer(60.0, reset_alarm_cooldown)
                        cooldown_timer.start()

                    detected_objects.add((x, y, w, h))

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        #Reset alarm flag and cooldown when no objects are detected.
        alarm_flag = False
        detected_objects.clear()

    #Return bounding box for humans.
    boxes_hog, weights_hog = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes_hog = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes_hog])

    for (xA, yA, xB, yB) in boxes_hog:
        #Display the box.
        cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 2)

    #Display the resulting frame.
    cv2.imshow('Output', frame)

    #Write the frame to the output video.
    out.write(frame.astype('uint8'))

    #Wait for 'q' key to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release capture and output.
cap.release()
out.release()

#Closing windows.
cv2.destroyAllWindows()
cv2.waitKey(1)