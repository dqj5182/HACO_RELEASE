import cv2
import numpy as np
from collections import defaultdict, deque

import mediapipe as mp


from lib.utils.vis_utils import draw_landmarks_on_image, draw_landmarks_on_image_simple


def smooth_bbox(prev_bbox, curr_bbox, alpha=0.8):
    if prev_bbox is None:
        return curr_bbox
    return [alpha * p + (1 - alpha) * c for p, c in zip(prev_bbox, curr_bbox)]


def smooth_contact_mask(prev_mask, curr_mask, alpha=0.8):
    if prev_mask is None:
        return curr_mask.astype(np.float32)
    return alpha * prev_mask + (1 - alpha) * curr_mask.astype(np.float32)


def remove_small_contact_components(contact_mask, faces, min_size=20):
    vertex_to_faces = defaultdict(list)
    for i, f in enumerate(faces):
        for v in f:
            vertex_to_faces[v].append(i)

    visited = np.zeros(len(contact_mask), dtype=bool)
    filtered_mask = np.zeros_like(contact_mask, dtype=bool)

    for v in range(len(contact_mask)):
        if visited[v] or not contact_mask[v]:
            continue

        queue = deque([v])
        component = []
        while queue:
            curr = queue.popleft()
            if visited[curr] or not contact_mask[curr]:
                continue
            visited[curr] = True
            component.append(curr)
            for f_idx in vertex_to_faces[curr]:
                for neighbor in faces[f_idx]:
                    if not visited[neighbor] and contact_mask[neighbor]:
                        queue.append(neighbor)

        if len(component) >= min_size:
            filtered_mask[component] = True

    return filtered_mask


def initialize_video_writer(output_path, fps, frame_size):
    tried_codecs = ['avc1', 'H264', 'X264', 'MJPG', 'mp4v'] # we recommend using 'MJPG'
    for codec in tried_codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if writer.isOpened():
            print(f"Using codec '{codec}' for {output_path}")
            return writer
        writer.release()
    raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")


def run_wilor_hand_detector(orig_img, detector):
    conf = 0.3
    IoU_threshold = 0.3

    detections = detector(orig_img, conf=conf, verbose=False, iou=IoU_threshold)[0]

    img_h, img_w, _ = orig_img.shape

    right_hand_bbox = [0, 0, img_w, img_h] # [x_min_expand, y_min_expand, bb_width_expand, bb_height_expand]
    best_conf = 0.

    # Find the most confident right hand
    for det in detections: 
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        Conf = det.boxes.conf.data.cpu().detach()[0].numpy().reshape(-1).astype(np.float16)
        Side = det.boxes.cls.data.cpu().detach()

        if (Side.item() == 1.) and (Conf.item() > best_conf):
            right_hand_bbox = [Bbox[0], Bbox[1], Bbox[2]-Bbox[0], Bbox[3]-Bbox[1]]
    
    return right_hand_bbox


def extract_frames_with_hand(cap, detector, detector_type='wilor'):
    frames_with_hand = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if detector_type == 'wilor':
            right_hand_bbox = run_wilor_hand_detector(orig_img, detector)
            _, right_hand_bbox = draw_landmarks_on_image_simple(orig_img.copy(), right_hand_bbox)
        elif detector_type == 'mediapipe':
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=orig_img)
            detection_result = detector.detect(mp_image)
            _, right_hand_bbox = draw_landmarks_on_image(orig_img.copy(), detection_result)

        if right_hand_bbox is not None:
            frames_with_hand.append((frame_idx, frame, right_hand_bbox))

        frame_idx += 1

    cap.release()
    return frames_with_hand


def find_longest_continuous_segment(frames_with_hand):
    longest_segment = []
    current_segment = []

    for i in range(len(frames_with_hand)):
        if i == 0 or frames_with_hand[i][0] == frames_with_hand[i - 1][0] + 1:
            current_segment.append(frames_with_hand[i])
        else:
            if len(current_segment) > len(longest_segment):
                longest_segment = current_segment
            current_segment = [frames_with_hand[i]]

    if len(current_segment) > len(longest_segment):
        longest_segment = current_segment

    return longest_segment