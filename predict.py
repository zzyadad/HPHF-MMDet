from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


model = YOLO(r"runs/detect/train/weights/best.pt")
model.predict(source=r"datasets/images/val", # 测试图像，指定可见光图像，会自动读取红外图像
              save=True
              )