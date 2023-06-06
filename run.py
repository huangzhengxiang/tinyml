from ultralytics import YOLO


if __name__=="__main__":
    # Load a model
    model = YOLO("yolov8n.yaml","ptq").load('yolov8n-voc.pt')  # build a new model from scratch
    # model = YOLO("yolov8n.yaml","qas").load('yolov8n.pt')  # build a new model from scratch
    # model = YOLO("yolov8n.yaml").load('yolov8n-relu.pt')  # build a new model from scratch
    # model = YOLO("yolov8n.yaml").load('yolov8n-voc.pt')  # build a new model from scratch
    # model = YOLO("yolov8n.yaml")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="coco128.yaml", epochs=1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # success = model.export(format="onnx")  # export the model to ONNX format