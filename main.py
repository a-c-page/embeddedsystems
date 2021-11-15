import cv2 as cv
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
file_name = 'labels.txt'

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

cap = cv.VideoCapture('traffic.mp4')

if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("ERROR")

while True:
    ret,frame = cap.read()
    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.45)

    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN

    if(len(ClassIndex) != 0):
        for ClassIndex, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassIndex <= 80):
                cv.rectangle(frame,boxes, (255,0,0), 2)
                cv.putText(frame,classLabels[ClassIndex - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,color=(0,255,0), thickness=3)

        cv.imshow('Tutorial', frame)
        
        if cv.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()