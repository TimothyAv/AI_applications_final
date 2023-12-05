# Object Detection System

This project implements an object detection system that uses three different models simultaneously to detect and alert for objects in a given video stream.
![image](https://github.com/TimothyAv/AI_applications_final/assets/83468895/61da33da-7f50-425e-977f-934824273a12)

## Models

### 1. Haar Cascade for Car Detection

![image](https://github.com/TimothyAv/AI_applications_final/assets/83468895/7570689d-78b2-4b8f-8d51-790438b4ab0a)

- **Model File**: `cars.xml`
- **Description**: Haar cascade classifier trained for detecting cars in images.
- **Usage**: Utilized to identify cars in the video stream.
- **Implementation**: Utilizes OpenCV's `CascadeClassifier` for car detection.

### 2. YOLO (You Only Look Once) for Animal Detection

![image](https://github.com/TimothyAv/AI_applications_final/assets/83468895/6597ebef-12d0-416e-b969-d66db41900c4)

- **Model Files**: `yolov3-tiny.weights`, `yolov3-tiny.cfg`
- **Description**: YOLO model for detecting various objects, focusing on animals.
- **Usage**: Detects animals in the video stream.
- **Implementation**: Utilizes OpenCV's `dnn` module for YOLO object detection.

### 3. HOG (Histogram of Oriented Gradients) for Human Detection

![image](https://github.com/TimothyAv/AI_applications_final/assets/83468895/58a2942d-0ceb-45ef-a67c-fd6fc9d024ec)

- **Model Configuration**: Default people detector in OpenCV
- **Description**: HOG-based detector for identifying humans.
- **Usage**: Detects humans in the video stream.
- **Implementation**: Utilizes OpenCV's `HOGDescriptor` for human detection.

## Dependencies

- **OpenCV**: Used for image and video processing, as well as implementing various computer vision models.
- **PIL (Pillow)**: Required for image-related operations.
- **playsound**: Used for playing alert sounds.
- **threading**: Utilized for multi-threading operations.
- **smtplib**: Used for sending email alerts.

## How to Run

1. Ensure all dependencies are installed. You can install them using `pip install -r requirements.txt`.
2. Place the video file in the project directory or update the video file path in the code.
3. Run the main script: `python object_detection.py`.
4. The system will detect and draw rectangles around cars, animals, and humans in the video stream.
5. An alert sound will play, and an email notification will be sent if an object of interest is detected.

## Configuration

- `cars.xml`: Haar cascade classifier configuration for car detection.
- `yolov3-tiny.weights`, `yolov3-tiny.cfg`: YOLO configuration and weights files.
- `alarmm.wav`: Alert sound file.

## Notes

- Adjust confidence levels, thresholds, and timers in the code as needed.
- Customize the list of `final_classes` to include specific classes you want to detect with YOLO.
- Set up a Google App Password for email notifications.
