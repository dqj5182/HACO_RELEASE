import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from lib.core.config import cfg, update_config
from lib.models.model import HACO
from lib.utils.human_models import mano
from lib.utils.contact_utils import get_contact_thres
from lib.utils.vis_utils import ContactRenderer, draw_landmarks_on_image, draw_landmarks_on_image_simple
from lib.utils.preprocessing import augmentation_contact
from lib.utils.demo_utils import remove_small_contact_components, run_wilor_hand_detector


parser = argparse.ArgumentParser(description='Demo HACO')
parser.add_argument('--backbone', type=str, default='hamer', choices=['hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'handoccnet', 'hrnet-w48', 'hrnet-w32', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')
parser.add_argument('--detector', type=str, default='wilor', choices=['wilor', 'mediapipe'], help='detector model')
parser.add_argument('--checkpoint', type=str, default='', help='model path for demo')
parser.add_argument('--input_path', type=str, default='asset/example_images', help='image path for demo')
args = parser.parse_args()


# Set device as CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Initialize directories
experiment_dir = 'experiments_demo_image'


# Load config
update_config(backbone_type=args.backbone, exp_dir=experiment_dir)


# Initialize renderer
contact_renderer = ContactRenderer()


# Load demo images
input_dir = args.input_path
images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


# Initialize MediaPipe HandLandmarker
if args.detector == 'wilor':
    from ultralytics import YOLO
    detector_path = f'data/base_data/demo_data/wilor_detector.pt'
    detector = YOLO(detector_path)
elif args.detector == 'mediapipe':
    base_options = BaseOptions(model_asset_path=cfg.MODEL.hand_landmarker_path)
    hand_options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(hand_options)
else:
    raise NotImplementedError(f"Unsupported detector: {args.detector}")


############# Model #############
model = HACO().to(device)
model.eval()
############# Model #############


# Load model checkpoint if provided
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


############################### Demo Loop ###############################
for i, frame_name in tqdm(enumerate(images), total=len(images)):
    print(f"Processing: {frame_name}")

    # Load and convert image
    frame_path = os.path.join(input_dir, frame_name)
    frame = cv2.imread(frame_path)
    orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_name_base = os.path.splitext(frame_name)[0]

    # Hand landmark detection
    if args.detector == 'wilor':
        right_hand_bbox = run_wilor_hand_detector(orig_img, detector)
        annotated_image, right_hand_bbox = draw_landmarks_on_image_simple(orig_img.copy(), right_hand_bbox)
    elif args.detector == 'mediapipe':
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=orig_img.copy())
        detection_result = detector.detect(mp_image)
        annotated_image, right_hand_bbox = draw_landmarks_on_image(orig_img.copy(), detection_result)
    else:
        raise NotImplementedError(f"Unsupported detector: {args.detector}")
    

    if right_hand_bbox is None:
        print(f"Skipping {frame_name} - no hand detected.")
        continue

    print(f"Frame {i}: Right hand bbox: {right_hand_bbox}")

    # Image preprocessing
    crop_img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), right_hand_bbox, 'test', enforce_flip=False)

    # Convert to model input format
    if args.backbone in ['handoccnet'] or 'resnet' in cfg.MODEL.backbone_type or 'hrnet' in cfg.MODEL.backbone_type:
        from torchvision import transforms
        img_tensor = transforms.ToTensor()(crop_img.astype(np.float32) / 255.0)
    elif args.backbone in ['hamer'] or 'vit' in cfg.MODEL.backbone_type:
        from torchvision.transforms import Normalize
        normalize = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
        img_tensor = crop_img.transpose(2, 0, 1) / 255.0
        img_tensor = normalize(torch.from_numpy(img_tensor)).float()
    else:
        raise NotImplementedError(f"Unsupported backbone: {args.backbone}")

    ############# Run model #############
    with torch.no_grad():
        outputs = model({'input': {'image': img_tensor[None].to(device)}}, mode="test")
    ############# Run model #############

    # Save result
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/detection', exist_ok=True)
    os.makedirs('outputs/crop_img', exist_ok=True)
    os.makedirs('outputs/contact', exist_ok=True)

    cv2.imwrite(f'outputs/detection/{frame_name_base}.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'outputs/crop_img/{frame_name_base}.png', crop_img[..., ::-1])

    eval_thres = get_contact_thres(args.backbone)
    contact_mask = (outputs['contact_out'].sigmoid()[0] > eval_thres).detach().cpu().numpy()
    contact_mask = remove_small_contact_components(contact_mask, faces=mano.watertight_face['right'], min_size=20)
    contact_rendered = contact_renderer.render_contact(crop_img[..., ::-1], contact_mask)
    cv2.imwrite(f'outputs/contact/{frame_name_base}.png', contact_rendered)
############################### Demo Loop ###############################