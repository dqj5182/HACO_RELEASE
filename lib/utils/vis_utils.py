import os
import cv2
import torch
import trimesh
import pyrender
import numpy as np
import matplotlib.cm as cm

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

os.environ["PYOPENGL_PLATFORM"] = "egl"

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.mano_utils import change_flat_hand_mean


# This function is modified from the function of DECO (https://github.com/sha2nkt/deco/blob/main/inference.py)
class ContactRenderer():
    def __init__(self):
        self.default_mesh_color = [130, 130, 130, 255]
        self.contact_mesh_color = [0, 255, 0, 255]

        with torch.no_grad():
            hand_pose = change_flat_hand_mean(np.zeros((48)), remove=True)
            mano_rest_out = mano.layer['right'](betas=torch.zeros((1, 10)), hand_pose=torch.from_numpy(hand_pose[None, 3:]).float(), global_orient=torch.zeros((1, 3)), transl=torch.zeros((1, 3)))
            self.hand_model_mano = trimesh.Trimesh(mano_rest_out.vertices[0], mano.watertight_face['right'])

    def render_image(self, scene, img_res, img=None, viewer=False):
        r = pyrender.OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img

    def create_scene(self, mesh, img, focal_length=5000, camera_center=250, img_res=500):
        # Setup the scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                            ambient_light=(0.3, 0.3, 0.3))
        # add mesh for camera
        camera_pose = np.eye(4)
        camera_rotation = np.eye(3, 3)
        camera_translation = np.array([0., 0, 2.5])
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation

        pyrencamera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center, cy=camera_center)
        scene.add(pyrencamera, pose=camera_pose)

        # create and add light
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            # out_mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        # add body mesh
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        mesh_images = []

        # resize input image to fit the mesh image height
        img_height = img_res
        img_width = int(img_height * img.shape[1] / img.shape[0])
        img = cv2.resize(img, (img_width, img_height))
        mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for sideview_angle in [0, 90, 180, 270]:
            out_mesh = mesh.copy()
            rot = trimesh.transformations.rotation_matrix(
                np.radians(sideview_angle), [0, 1, 0])
            out_mesh.apply_transform(rot)
            out_mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)
            mesh_pose = np.eye(4)
            scene.add(out_mesh, pose=mesh_pose, name='mesh')

            output_img = self.render_image(scene, img_res)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2RGB)
            mesh_images.append(output_img)

            # delete the previous mesh
            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        # show upside down view
        for topview_angle in [90, 270]:
            out_mesh = mesh.copy()
            rot = trimesh.transformations.rotation_matrix(
                np.radians(topview_angle), [1, 0, 0])
            out_mesh.apply_transform(rot)
            out_mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)
            mesh_pose = np.eye(4)
            scene.add(out_mesh, pose=mesh_pose, name='mesh')

            output_img = self.render_image(scene, img_res)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2RGB)
            mesh_images.append(output_img)

            # delete the previous mesh
            prev_mesh = scene.get_nodes(name='mesh').pop()
            scene.remove_node(prev_mesh)

        # stack images
        IMG = np.hstack(mesh_images)
        return IMG

    def create_scene_demo(self, mesh, img, focal_length=5000, camera_center=250, img_res=500):
        # Setup the scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                            ambient_light=(0.3, 0.3, 0.3))
        
        # Camera
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0., 0, 2.5])
        pyrencamera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center, cy=camera_center)
        scene.add(pyrencamera, pose=camera_pose)

        # Lighting
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        # Material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        mesh_images = []

        # Resize input image
        img_height = img_res
        img_width = int(img_height * img.shape[1] / img.shape[0])
        img = cv2.resize(img, (img_width, img_height))
        mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Top views only (X-axis rotations), then rotate 90° clockwise
        for topview_angle in [90, 270]:
            out_mesh = mesh.copy()

            # Rotate around X-axis
            rot = trimesh.transformations.rotation_matrix(
                np.radians(topview_angle), [1, 0, 0])
            out_mesh.apply_transform(rot)

            # Move mesh to the right (positive X-axis) and assign label
            if topview_angle == 90:
                right_shift = np.array([-0.02, 0.03, 0.0])  # Dorsal view
                label = "Dorsal"
            elif topview_angle == 270:
                right_shift = np.array([-0.02, -0.025, 0.0])  # Palmar view
                label = "Palmar"
            out_mesh.apply_translation(right_shift)

            # Create pyrender mesh and add to scene
            mesh_node = pyrender.Mesh.from_trimesh(out_mesh, material=material)
            mesh_pose = np.eye(4)
            scene.add(mesh_node, pose=mesh_pose, name='mesh')

            # Render the scene
            output_img = self.render_image(scene, img_res)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGBA2RGB)

            # Rotate 90 degrees clockwise
            output_img = cv2.rotate(output_img, cv2.ROTATE_90_CLOCKWISE)

            # Write label directly on the image (bottom center)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            if topview_angle == 90:
                text_x_move = 44
            elif topview_angle == 270:
                text_x_move = -34
            text_x = (output_img.shape[1] - text_size[0]) // 2 + text_x_move
            text_y = output_img.shape[0] - 25  # 10px above bottom
            cv2.putText(output_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            mesh_images.append(output_img)

            # Remove the mesh node
            scene.remove_node(scene.get_nodes(name='mesh').pop())

        # Stack images horizontally
        IMG = np.hstack(mesh_images)
        return IMG

    def render_contact(self, img, contact, mode='test'):   
        vis_contact = contact == 1.

        for vert in range(self.hand_model_mano.visual.vertex_colors.shape[0]):
            self.hand_model_mano.visual.vertex_colors[vert] = self.default_mesh_color
        self.hand_model_mano.visual.vertex_colors[vis_contact] = self.contact_mesh_color

        img = cv2.resize(img.copy(), cfg.MODEL.input_img_shape, cv2.INTER_CUBIC)

        if mode == 'demo':
            rend = self.create_scene_demo(self.hand_model_mano, img[..., ::-1].astype(np.uint8))
        else:
            rend = self.create_scene(self.hand_model_mano, img[..., ::-1].astype(np.uint8))
        return rend



# This function is for demo code with mediapipe
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    right_hand_bbox = None

    best_score = -1.0
    best_idx = -1

    # Step 1: Find the index of the most confident right hand
    for idx in range(len(hand_landmarks_list)):
        handedness = handedness_list[idx][0]  # get Classification result
        if handedness.category_name == "Right" and handedness.score > best_score:
            best_score = handedness.score
            best_idx = idx

    # Step 2: If a right hand was found, draw it and extract bbox
    if best_idx != -1:
        hand_landmarks = hand_landmarks_list[best_idx]
        handedness = handedness_list[best_idx]

        # Convert landmarks to protobuf for drawing
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Compute bbox
        height, width, _ = annotated_image.shape
        x_coords = [lm.x * width for lm in hand_landmarks]
        y_coords = [lm.y * height for lm in hand_landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        bb_c_x, bb_c_y = (x_min+x_max)/2, (y_min+y_max)/2
        bb_width, bb_height = x_max-x_min, y_max-y_min

        expand_ratio = cfg.DATASET.ho_big_bbox_expand_ratio

        bb_width_expand, bb_height_expand = expand_ratio * bb_width, expand_ratio * bb_height
        x_min_expand, y_min_expand = bb_c_x - 0.5 * bb_width_expand, bb_c_y - 0.5 * bb_height_expand
        
        right_hand_bbox = [x_min_expand, y_min_expand, bb_width_expand, bb_height_expand]

        # Draw bbox and label
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(annotated_image, "Right Hand", (x_min, y_min - MARGIN),
                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image, right_hand_bbox