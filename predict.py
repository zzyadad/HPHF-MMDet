from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')


model =RTDETR(r"")
model.predict(source=r"", 
              save=True
              )
