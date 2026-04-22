from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')
# python -m torch.distributed.launch  --nproc_per_node=4   --master_port=29000    train.py  

model = RTDETR("")  # 多模态模型

model.train(data='',  # 训练参数均可以重新设置
                        epochs=1, 
                        imgsz=640, 
                        workers=8, 
                        batch=1,
                        device=0,
                        optimizer='AdamW',
                        amp=False,
                                           lr0=0.0001,   # (float) 初始学习率（例如SGD=1E-2, Adam=1E-3）
lrf=0.01, # (float) 最终学习率（lr0 * lrf）

)
#                          lr0=0.001,  # (float) 初始学习率（例如SGD=1E-2, Adam=1E-3）
# lrf=0.01, # (float) 最终学习率（lr0 * lrf）
# momentum=0.937, # (float) SGD动量/Adam beta1
# weight_decay=0.0005, # (float) 优化器权重衰减5e-4
# warmup_epochs=3.0, # (float) 预热轮数（小数可）
# warmup_momentum=0.8, # (float) 预热初始动量
# warmup_bias_lr=0.1, # (float) 预热初始偏差学习率
# box=7.5, # (float) 盒子损失增益
# cls=0.5, # (float) 分类损失增益（与像素比例）
# dfl=1.5, # (float) dfl损失增益
# pose=12.0, # (float) 姿态损失增益
# kobj=1.0, # (float) 关键点对象损失增益
# label_smoothing=0.0, # (float) 标签平滑（比例）
# nbs=64, # (int) 名义批次大小
# hsv_h=0.015, # (float) 图像HSV-色调增强（比例）
# hsv_s=0.7, # (float) 图像HSV-饱和度增强（比例）
# hsv_v=0.4, # (float) 图像HSV-值增强（比例）
# degrees=0.0, # (float) 图像旋转（+/-度）
# translate=0.1, # (float) 图像平移（+/-比例）
# scale=0.5, # (float) 图像缩放（+/-增益）
# shear=0.0, # (float) 图像剪切（+/-度）
# perspective=0.0, # (float) 图像透视（+/-比例），范围0-0.001
# flipud=0.0, # (float) 图像上下翻转（概率）
# fliplr=0.5, # (float) 图像左右翻转（概率）
# bgr=0.0, # (float) 图像通道BGR（概率）
# mosaic=1.0, # (float) 图像马赛克（概率）
# mixup=0.0, # (float) 图像混合（概率）                                                   
# copy_paste=0.0, # (float) 分割复制粘贴（概率）
# copy_paste_mode="flip", # (str) 复制粘贴模式 (flip, mixup)
# auto_augment='randaugment', # (str) 用于分类的自动增强策略（randaugment、autoaugment、augmix）
# erasing=0.4, # (float) 分类训练过程中随机擦除概率（0-0.9），0表示不擦除，必须小于1.0。
# crop_fraction=1.0 # (float) 用于分类的图像裁剪分数（0.1-1），1.0表示没有裁剪，必须大于0。
 
#                         ) 

# lr0=0.01,  # (float) 初始学习率（例如SGD=1E-2, Adam=1E-3）
# lrf=0.01, # (float) 最终学习率（lr0 * lrf）
# momentum=0.937, # (float) SGD动量/Adam beta1
# weight_decay=0.0005, # (float) 优化器权重衰减5e-4
# warmup_epochs=3.0, # (float) 预热轮数（小数可）
# warmup_momentum=0.8, # (float) 预热初始动量
# warmup_bias_lr=0.1, # (float) 预热初始偏差学习率
# box=7.5, # (float) 盒子损失增益
# cls=0.5, # (float) 分类损失增益（与像素比例）
# dfl=1.5, # (float) dfl损失增益
# pose=12.0, # (float) 姿态损失增益
# kobj=1.0, # (float) 关键点对象损失增益
# label_smoothing=0.0, # (float) 标签平滑（比例）
# nbs=64, # (int) 名义批次大小
# hsv_h=0.015, # (float) 图像HSV-色调增强（比例）
# hsv_s=0.7, # (float) 图像HSV-饱和度增强（比例）
# hsv_v=0.4, # (float) 图像HSV-值增强（比例）
# degrees=0.0, # (float) 图像旋转（+/-度）
# translate=0.1, # (float) 图像平移（+/-比例）
# scale=0.5, # (float) 图像缩放（+/-增益）
# shear=0.0, # (float) 图像剪切（+/-度）
# perspective=0.0, # (float) 图像透视（+/-比例），范围0-0.001
# flipud=0.0, # (float) 图像上下翻转（概率）
# fliplr=0.5, # (float) 图像左右翻转（概率）
# bgr=0.0, # (float) 图像通道BGR（概率）
# mosaic=1.0, # (float) 图像马赛克（概率）
# mixup=0.0, # (float) 图像混合（概率）                                                   
# copy_paste=0.0, # (float) 分割复制粘贴（概率）
# copy_paste_mode="flip", # (str) 复制粘贴模式 (flip, mixup)
# auto_augment: randaugment # (str) 用于分类的自动增强策略（randaugment、autoaugment、augmix）
# erasing=0.4, # (float) 分类训练过程中随机擦除概率（0-0.9），0表示不擦除，必须小于1.0。
# crop_fraction=1.0 # (float) 用于分类的图像裁剪分数（0.1-1），1.0表示没有裁剪，必须大于0。



                       # lr0=0.0001,  # (float) 初始学习率（例如SGD=1E-2, Adam=1E-3）
                        # lrf=0.001, # (float) 最终学习率（lr0 * lrf）
                        # momentum=0.900, # (float) SGD动量/Adam beta1
                        # weight_decay=0.0001, # (float) 优化器权重衰减5e-4
                        # warmup_epochs=3.0, # (float) 预热轮数（小数可）
                        # warmup_momentum=0.8, # (float) 预热初始动量
                        # warmup_bias_lr=0.1, # (float) 预热初始偏差学习率
                        # box=7.5, # (float) 盒子损失增益
                        # cls=0.5, # (float) 分类损失增益（与像素比例）
                        # dfl=1.5, # (float) dfl损失增益
                        # pose=12.0, # (float) 姿态损失增益
                        # kobj=1.0, # (float) 关键点对象损失增益
                        # label_smoothing=0.0, # (float) 标签平滑（比例）
                        # nbs=64, # (int) 名义批次大小
                        # hsv_h=0.015, # (float) 图像HSV-色调增强（比例）
                        # hsv_s=0.7, # (float) 图像HSV-饱和度增强（比例）
                        # hsv_v=0.4, # (float) 图像HSV-值增强（比例）
                        # degrees=0.0, # (float) 图像旋转（+/-度）
                        # translate=0.1, # (float) 图像平移（+/-比例）
                        # scale=0.5, # (float) 图像缩放（+/-增益）
                        # shear=0.0, # (float) 图像剪切（+/-度）
                        # perspective=0.0, # (float) 图像透视（+/-比例），范围0-0.001
                        # flipud=0.0, # (float) 图像上下翻转（概率）
                        # fliplr=0.5, # (float) 图像左右翻转（概率）
                        # bgr=0.0, # (float) 图像通道BGR（概率）
                        # mosaic=0.0, # (float) 图像马赛克（概率）
                        # mixup=0.0, # (float) 图像混合（概率）                                                   
                        # copy_paste=0.0, # (float) 分割复制粘贴（概率）
                        # copy_paste_mode="flip", # (str) 复制粘贴模式 (flip, mixup)
                        # auto_augment='randaugment', # (str) 用于分类的自动增强策略（randaugment、autoaugment、augmix）
                        # erasing=0.4, # (float) 分类训练过程中随机擦除概率（0-0.9），0表示不擦除，必须小于1.0。
                        # crop_fraction=1.0 # (float) 用于分类的图像裁剪分数（0.1-1），1.0表示没有裁剪，必须大于0。   