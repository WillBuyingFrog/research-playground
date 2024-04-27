import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

import argparse
from collections import OrderedDict
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, device=None):
        
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes, box_detections_per_img=300)
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None
        self.device = device

    def detect(self, images):

        # if self.device is not None:
        #     print(f'Image tensor type {type(images)}')
        #     images = images.to(self.device)
        #     print(f'Using device {self.device}')
        #     print(f'Image tensor type {type(images)}')
            
        detections = self(images)[0]

        return detections['boxes'].detach(), detections['scores'].detach(), detections['labels'].detach()
    
    def load_image(self, images):
        if self.device is not None:
            images = images.to(device)
        elif torch.cuda.is_available():
            images = images.cuda()

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
    
# TODO 实现函数：
#   输入：检测到的中央凹部分目标原始boxes，中央凹区域在原始图像中的tlwh，原图长宽缩小比例（默认长宽缩小比例一致）
#   输出：中央凹区域图像中检测到的所有目标的boxes，
#       要求所有boxes均已经转换到缩小后原图的坐标位置


def get_processed_boxes(fovea_boxes, fovea_pos, fovea_scale=[3.0, 3.0]):

    processed_boxes = []

    for fovea_box in fovea_boxes:
        
        processed_box = np.zeros_like(fovea_box)
        processed_box[0] = (fovea_pos[0] + fovea_box[0]) / fovea_scale[0]
        processed_box[1] = (fovea_pos[1] + fovea_box[1]) / fovea_scale[1]
        processed_box[2] = (fovea_pos[0] + fovea_box[2]) / fovea_scale[0]
        processed_box[3] = (fovea_pos[1] + fovea_box[3]) / fovea_scale[1]
        processed_boxes.append(processed_box)
    
    # 将origin_boxes转成pytorch tensor
    processed_boxes = torch.tensor(processed_boxes, dtype=torch.float32)
    
    return processed_boxes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Faster R-CNN Test')

    parser.add_argument('--state_dict', type=str, default='/home/aa/frog/mot-dbt/tracking_wo_bnw/output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model')
    parser.add_argument('--img_path', type=str, default='/home/aa/frog/mot-dbt/research-playground/pics')
    parser.add_argument('--results_path', type=str, default='/home/aa/frog/mot-dbt/research-playground/results')
    parser.add_argument('--check_mps', action='store_true')

    args = parser.parse_args()

    

    if args.check_mps:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built() and (not torch.cuda.is_available()):
            print(f'MPS available')
            device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using cuda')
        device = torch.device("cuda")
    else:
        device = None

    obj_detect = FRCNN_FPN(num_classes=2, device=device)
    obj_detect.to(device)

    

    obj_detect_state_dict = torch.load(args.state_dict, map_location=lambda storage, loc:storage)
    if 'model' in obj_detect_state_dict:
        obj_detect_state_dict = obj_detect_state_dict['model']
    
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()

    

    if args.check_mps:
        if device:
            print('Using MPS for Faster RCNN')

    # Transform
    # 根据Tracktor的MOTSequence实现，这里应当使用ToTensor

    transform = ToTensor()


    for file in os.listdir(args.img_path):

        if file.endswith('.jpg'):
            img_path = os.path.join(args.img_path, file)
            img_raw = Image.open(img_path).convert('RGB')
            img = transform(img_raw)

            img_list = torch.stack([img], dim=0).to(device)
            # 获取img_raw的长宽
            img_h, img_w = img_raw.size


            print(f'Processing {img_path}')
            
            boxes, scores, labels = obj_detect.detect(img_list)
            # 将boxes挪到cpu上
            boxes = boxes.cpu()
            print(f'Finished detecting image, {len(boxes)} objects detected')

            origin_img_h, origin_img_w = img_h * 2, img_w * 2
            print(f'Origin image size: {origin_img_h} x {origin_img_w}')
            print(f'Foveal image size: {img_h} x {img_w}')
            
            fovea_scale = [5.0, 5.0]


            # 模拟中央凹区域检测锚框转回处理后图像锚框的流程
            # 假设这里处理的每张图片都是一张长宽均为其两倍的原始图P的正中央1/4部分，在这里我们默认图像P中除了正中央1/4部分的内容为img，剩下的地方均为纯白色
            # 并假设处理后图像为P长宽各缩小5倍，获得处理后图像Q

            # 计算P中，中央凹区域的box
            fovea_pos = [origin_img_h / 4, origin_img_w / 4, origin_img_h / 2, origin_img_w / 2]

            # 计算Q中box的对应坐标
            processed_boxes = get_processed_boxes(boxes, fovea_pos, fovea_scale)

            # for index, _ in enumerate(boxes):
            #     print(f'Origin box: {boxes[index]}, Processed box: {processed_boxes[index]}')
            
            # 绘图，将原始的boxes画到img上，保存为 {原始文件名}_detect.jpg
            # 将处理后的boxes画到P上，保存为 {原始文件名}_processed.jpg

            img_detect = img_raw.copy()
            
            # 创建img_processed变量，为上述注释中所述的图像Q

            # 首先，我们需要创建一个与img相同大小的纯白色图像作为基础
            img_processed = Image.new('RGB', (origin_img_h, origin_img_w), color='white')
            # 然后将img复制到img_processed的中央位置
            fovea_box = (img_h // 2, img_w // 2)
            print(f'fovea_box is {fovea_box}')
            Image.Image.paste(img_processed, img_raw, fovea_box)
            # 最后，将img_processed缩小到原大小的1/5
            img_processed = img_processed.resize((int(origin_img_h / fovea_scale[0]),
                                                   int(origin_img_w / fovea_scale[1])), Image.LANCZOS)
            
            draw = ImageDraw.Draw(img_detect)
            print('Fovea image boxes:')
            for box in boxes:
                # 将box的坐标转换为左上角和右下角的形式
                top_left = (box[0], box[1])
                bottom_right = (box[2], box[3])
                print(f'top_left is {top_left}, bottom_right is {bottom_right}')
                # 绘制矩形框，颜色为橙色
                draw.rectangle([top_left, bottom_right], outline=(255, 165, 0), width=2)

            draw_processed = ImageDraw.Draw(img_processed)
            print('Processed image boxes:')
            for box in processed_boxes:
                # 为img_processed绘制出矩形锚框
                top_left = (box[0], box[1])
                bottom_right = (box[2], box[3])
                print(f'top_left is {top_left}, bottom_right is {bottom_right}')
                # 绘制矩形框，颜色为橙色
                draw_processed.rectangle([top_left, bottom_right], outline=(255, 165, 0), width=2)
                
            # 获取图像的原文件名
            base_filename = os.path.splitext(file)[0]
            results_path = args.results_path
            # 保存img_detect到results文件夹，文件名为{图像原文件名}_detect.jpg
            img_detect_filename = f"{base_filename}_detect.jpg"
            img_detect.save(os.path.join(results_path, img_detect_filename))

            # 保存img_processed到results文件夹，文件名为{图像原文件名}_processed.jpg
            img_processed_filename = f"{base_filename}_processed.jpg"
            img_processed.save(os.path.join(results_path, img_processed_filename))

            