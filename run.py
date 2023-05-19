from ultralytics import YOLO

# Load a model
# model = YOLO('runs/detect/VOC/weights/best.pt')
model = YOLO('yolov8n.yaml')  # 'runs/detect/train13/weights/best.pt'
# print(model.model.load_state_dict(torch.load('runs/detect/train13/weights/best.pt')['model']))
# Use the model
model.train(data="coco128.yaml", epochs=100)  # train the model
metrics = model.val(data='coco128.yaml')  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format