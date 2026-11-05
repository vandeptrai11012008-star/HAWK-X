
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect people in real time using your webcam
for result in model.predict(
    source=0,
    classes=[0],
    conf=0.4,
    show=True,
    save=False,
    stream=True
):
    pass
