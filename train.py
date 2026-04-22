from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')
# python -m torch.distributed.launch  --nproc_per_node=4   --master_port=29000    train.py  

model = RTDETR("")  

model.train(data='', 
                        epochs=1, 
                        imgsz=640, 
                        workers=8, 
                        batch=1,
                        device=0,
                        optimizer='AdamW',
                        amp=False,
                                           lr0=0.0001,  
lrf=0.01,

)
