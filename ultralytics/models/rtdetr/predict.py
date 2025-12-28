# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):

        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []  # 可见光结果
        ir_results = []  # 红外结果
        
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)
            idx = max_score.squeeze(-1) > self.args.conf
            
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
                
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
            
            # 处理可见光图像（后3通道）
            vis_img = orig_img[..., 3:] if orig_img.shape[-1] >= 6 else orig_img
            oh, ow = vis_img.shape[:2]
            pred_vis = pred.clone()
            pred_vis[..., [0, 2]] *= ow
            pred_vis[..., [1, 3]] *= oh
            results.append(Results(vis_img, path=img_path, names=self.model.names, boxes=pred_vis))
            
            # 处理红外图像（前3通道）
            if orig_img.shape[-1] >= 6:
                ir_img = orig_img[..., :3]
                ir_oh, ir_ow = ir_img.shape[:2]
                pred_ir = pred.clone()
                pred_ir[..., [0, 2]] *= ir_ow
                pred_ir[..., [1, 3]] *= ir_oh
                
                # 构建红外图像路径
                ir_path = img_path.split('images')
                ir_path = str(ir_path[0] + 'image' + ir_path[1]) if len(ir_path) > 1 else img_path
                ir_results.append(Results(ir_img, path=ir_path, names=self.model.names, boxes=pred_ir))
        
        return results, ir_results

    def pre_transform(self, im):

        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]


# # Ultralytics YOLO 🚀, AGPL-3.0 license

# import torch

# from ultralytics.data.augment import LetterBox
# from ultralytics.engine.predictor import BasePredictor
# from ultralytics.engine.results import Results
# from ultralytics.utils import ops


# class RTDETRPredictor(BasePredictor):
#     """
#     RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions using
#     Baidu's RT-DETR model.

#     This class leverages the power of Vision Transformers to provide real-time object detection while maintaining
#     high accuracy. It supports key features like efficient hybrid encoding and IoU-aware query selection.

#     Example:
#         ```python
#         from ultralytics.utils import ASSETS
#         from ultralytics.models.rtdetr import RTDETRPredictor

#         args = dict(model="rtdetr-l.pt", source=ASSETS)
#         predictor = RTDETRPredictor(overrides=args)
#         predictor.predict_cli()
#         ```

#     Attributes:
#         imgsz (int): Image size for inference (must be square and scale-filled).
#         args (dict): Argument overrides for the predictor.
#     """

#     def postprocess(self, preds, img, orig_imgs):
#         """
#         Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

#         The method filters detections based on confidence and class if specified in `self.args`.

#         Args:
#             preds (list): List of [predictions, extra] from the model.
#             img (torch.Tensor): Processed input images.
#             orig_imgs (list or torch.Tensor): Original, unprocessed images.

#         Returns:
#             (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
#                 and class labels.
#         """
#         if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
#             preds = [preds, None]

#         nd = preds[0].shape[-1]
#         bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

#         if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
#             orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

#         results = []
#         for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 4)
#             bbox = ops.xywh2xyxy(bbox)
#             max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
#             idx = max_score.squeeze(-1) > self.args.conf  # (300, )
#             if self.args.classes is not None:
#                 idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
#             pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # filter
#             oh, ow = orig_img.shape[:2]
#             pred[..., [0, 2]] *= ow
#             pred[..., [1, 3]] *= oh
#             results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
#         return results

#     def pre_transform(self, im):
#         """
#         Pre-transforms the input images before feeding them into the model for inference. The input images are
#         letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scaleFilled.

#         Args:
#             im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

#         Returns:
#             (list): List of pre-transformed images ready for model inference.
#         """
#         letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
#         return [letterbox(image=x) for x in im]
