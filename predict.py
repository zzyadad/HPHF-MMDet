from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')


model =RTDETR(r"")
model.predict(source=r"", # 测试图像，指定可见光图像，会自动读取红外图像
              save=True
              )