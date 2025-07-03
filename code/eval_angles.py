import os
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2

def get_obb_params_from_8_points(normalized_points, img_width, img_height):
    points_abs = np.array([
        normalized_points[0] * img_width, normalized_points[1] * img_height,
        normalized_points[2] * img_width, normalized_points[3] * img_height,
        normalized_points[4] * img_width, normalized_points[5] * img_height,
        normalized_points[6] * img_width, normalized_points[7] * img_height
    ]).reshape(-1, 2)

    rect = cv2.minAreaRect(points_abs.astype(np.float32))
    center_x, center_y = rect[0]
    width, height = rect[1]
    angle = rect[2]

    if width < height:
        width, height = height, width
        angle += 90

    return center_x, center_y, width, height, angle

def calculate_shortest_angle_diff(angle1_deg, angle2_deg):
    diff = abs(angle1_deg - angle2_deg)
    diff = diff % 180
    if diff > 90:
        diff = 180 - diff
    return diff

if __name__ == '__main__':
    model_path = '/home/krishnapole/all/gray-sci-labs/runs/obb/train4/weights/best.pt'
    images_directory = '/home/krishnapole/all/gray-sci-labs/datasets/obb/test/images/'
    labels_directory = '/home/krishnapole/all/gray-sci-labs/datasets/obb/test/labels/'
    object_classes = ['house', 'tennis court']

    model = YOLO(model_path)

    all_angle_errors = []
    
    image_files = [f for f in os.listdir(images_directory) if f.lower().endswith(('.jpg', '.png'))]
    image_files.sort()
    
    print(f"Evaluating {len(image_files)} test images for angle accuracy...")

    for image_filename in image_files:
        image_full_path = os.path.join(images_directory, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_full_path = os.path.join(labels_directory, label_filename)

        if not os.path.exists(label_full_path):
            continue

        try:
            with Image.open(image_full_path) as img:
                img_width, img_height = img.size
        except Exception:
            continue

        ground_truths = []
        try:
            with open(label_full_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) != 9:
                        continue
                    
                    class_id = int(parts[0])
                    cx, cy, w, h, angle_deg = get_obb_params_from_8_points(parts[1:], img_width, img_height)
                    
                    ground_truths.append({
                        'class_id': class_id, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle_deg': angle_deg
                    })
        except Exception:
            continue

        model_predictions = model.predict(image_full_path, verbose=False, conf=0.25)

        matched_gt_indices = set()
        for prediction_result in model_predictions:
            if prediction_result.obb is None:
                continue
            
            for detected_object_data in prediction_result.obb.data:
                det_list = detected_object_data.tolist()
                
                pred_cx, pred_cy, pred_w, pred_h, pred_angle_rad, pred_confidence, pred_class_id_float = det_list
                
                pred_class_id = int(pred_class_id_float)
                pred_angle_deg = np.degrees(pred_angle_rad)

                best_gt_match = None
                min_distance_to_gt = float('inf')

                for i, gt_object in enumerate(ground_truths):
                    if i in matched_gt_indices:
                        continue
                    
                    if gt_object['class_id'] == pred_class_id:
                        distance_between_centers = np.sqrt((pred_cx - gt_object['cx'])**2 + (pred_cy - gt_object['cy'])**2)
                        
                        gt_object_diagonal = np.sqrt(gt_object['w']**2 + gt_object['h']**2)
                        
                        if distance_between_centers < min_distance_to_gt and distance_between_centers < gt_object_diagonal * 0.5:
                            min_distance_to_gt = distance_between_centers
                            best_gt_match = (i, gt_object)

                if best_gt_match:
                    gt_idx, matched_gt_object = best_gt_match
                    
                    angle_error = calculate_shortest_angle_diff(pred_angle_deg, matched_gt_object['angle_deg'])
                    all_angle_errors.append(angle_error)
                    matched_gt_indices.add(gt_idx)

    if all_angle_errors:
        mean_absolute_angle_error = np.mean(all_angle_errors)
        print(f"\nAngle accuracy results")
        print(f"Total matched objects evaluated for angle: {len(all_angle_errors)}")
        print(f"Mean Absolute angle error: {mean_absolute_angle_error:.2f} degrees")