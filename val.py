import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('runs/detect/exp/weights/best.pt')
    model.val(data='data.yaml',
                imgsz=512,
                batch=8,
                split='val',
                workers=8,
                device='0',
                )
 
