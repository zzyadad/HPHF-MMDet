from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

model = YOLO(r"improve_multimodal/yolov13/yolov13-earlyfusion.yaml")  # 多模态模型
model.train(data=r"data.yaml",  # 数据集路径
            batch=8,
            imgsz=512,
            epochs=1,
            amp=False,
            workers=8,
            optimizer='SGD',
            close_mosaic=0,
            device='0'
            )  
