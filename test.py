from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import Detection

model = YOLO("D:\Perkuliahan\Semester 7\Deep Learning\webapp\webapp2\models\sest.pt")
model.predict(source="0", show=True, conf=0.5)