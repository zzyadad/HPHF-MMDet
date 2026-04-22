from ultralytics.models import RTDETR
 
 
if __name__ == '__main__':
    model = RTDETR(model="")
    #with open("/home/s-zhangzy/RT-DETR/yolo-multimodal_new/val_model.txt", "w", encoding="utf-8") as f:
        #print(model, file=f)
    model.val(data='',split='val',batch=3, device='2', imgsz=640, workers=8,conf=0.5)


