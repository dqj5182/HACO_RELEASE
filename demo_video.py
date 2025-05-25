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
from lib.utils.vis_utils import ContactRenderer, draw_landmarks_on_image
from lib.utils.preprocessing import augmentation_contact
from lib.utils.demo_utils import smooth_bbox, smooth_contact_mask, remove_small_contact_components, initialize_video_writer, extract_frames_with_hand, find_longest_continuous_segment


parser = argparse.ArgumentParser(description='Demo HACO')
parser.add_argument('--backbone', type=str, default='hamer', choices=['hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'handoccnet', 'hrnet-w48', 'hrnet-w32', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')
parser.add_argument('--checkpoint', type=str, default='', help='model path for demo')
parser.add_argument('--input_path', type=str, default='asset/example_videos', help='video path for demo')
args = parser.parse_args()


# Set device as CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Initialize directories
experiment_dir = 'experiments_demo_video'


# Load config
update_config(backbone_type=args.backbone, exp_dir=experiment_dir)


# Initialize renderer
contact_renderer = ContactRenderer()


# Load demo videos
input_dir = args.input_path
video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]


# Initialize MediaPipe HandLandmarker
base_options = BaseOptions(model_asset_path=cfg.MODEL.hand_landmarker_path)
hand_options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(hand_options)


############# Model #############
model = HACO().to(device)
model.eval()
############# Model #############


# Load model checkpoint if provided
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


############################### Demo Loop ###############################
for i, video_name in tqdm(enumerate(video_files), total=len(video_files)):
    print(f"Processing: {video_name}")

    # Organize input and output path
    video_path = os.path.join(input_dir, video_name)
    os.makedirs("outputs_video", exist_ok=True)
    output_path = os.path.join("outputs_video", f"{os.path.splitext(video_name)[0]}_out.mp4")

    # Load and convert video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if fps == 0 or np.isnan(fps) else fps

    # Extract meaningful video segment
    frames_with_hand = extract_frames_with_hand(cap, detector)
    longest_segment = find_longest_continuous_segment(frames_with_hand)

    if not longest_segment:
        print(f"No hand detected in any continuous segment for {video_name}")
        continue

    writer = None
    smoothed_bbox = None
    smoothed_contact = None

    for _, frame, bbox in longest_segment:
        # Image preprocessing
        smoothed_bbox = smooth_bbox(smoothed_bbox, bbox, alpha=0.8)
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), smoothed_bbox, 'test', enforce_flip=False, bkg_color='white')

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
        eval_thres = get_contact_thres(args.backbone)
        raw_contact = (outputs['contact_out'][0] > eval_thres).detach().cpu().numpy()
        smoothed_contact = smooth_contact_mask(smoothed_contact, raw_contact, alpha=0.8)
        contact_mask = smoothed_contact > 0.5
        contact_mask = remove_small_contact_components(contact_mask, faces=mano.watertight_face['right'], min_size=20)
        contact_rendered = contact_renderer.render_contact(crop_img, contact_mask, mode='demo')

        if writer is None:
            ch, cw = contact_rendered.shape[:2]
            writer = initialize_video_writer(output_path, fps, (cw, ch))

        writer.write(cv2.cvtColor(contact_rendered, cv2.COLOR_RGB2BGR))

    if writer:
        writer.release()
############################### Demo Loop ###############################