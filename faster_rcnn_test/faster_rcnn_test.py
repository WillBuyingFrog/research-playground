import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.transforms import ToTensor

import argparse
from collections import OrderedDict
from PIL import Image
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

        if self.device is not None:
            images = images.to(self.device)
        if torch.cuda.is_available():
            images = images.cuda()
            
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
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Faster R-CNN Test')

    parser.add_argument('--state_dict', type=str, default='/Users/frog_wch/playground/Research/Weights/model_epoch_27.model')
    parser.add_argument('--img_path', type=str, default='/Users/frog_wch/playground/Research/Repos/research-playground/results')
    parser.add_argument('--check_mps', action='store_true')

    args = parser.parse_args()

    

    if args.check_mps:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built() and (not torch.cuda.is_available()):
            print(f'MPS available')
            device = torch.device("mps")
    else:
        device = None

    obj_detect = FRCNN_FPN(num_classes=2, device=device)

    

    obj_detect_state_dict = torch.load(args.state_dict, map_location=lambda storage, loc:storage)
    if 'model' in obj_detect_state_dict:
        obj_detect_state_dict = obj_detect_state_dict['model']
    
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()

    if torch.cuda.is_available():
        obj_detect.cuda()

    if args.check_mps:
        if device:
            obj_detect = obj_detect.to(device)
            print('Using MPS for Faster RCNN')

    # Transform
    # 根据Tracktor的MOTSequence实现，这里应当使用ToTensor

    transform = ToTensor()


    for file in os.listdir(args.img_path):

        if file.endswith('.jpg'):
            img_path = os.path.join(args.img_path, file)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)

            img_list = torch.stack([img], dim=0)
            

            print(f'Processing {img_path}')
            
            boxes, scores, labels = obj_detect.detect(img_list)
            # print(boxes, scores, labels)
            print(f'Finished detecting image, {len(boxes)} objects detected')