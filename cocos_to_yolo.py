import json
from pathlib import Path
from tqdm import tqdm
import shutil
import os

def coco_to_yolo(coco_json_path, image_dir, output_root_dir, class_names):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    output_images_dir = output_root_dir / 'images'
    output_labels_dir = output_root_dir / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    category_id_to_yolo_id = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    }

    img_map = {img['id']: img for img in coco_data['images']}
    processed_images = set()

    for annotation in tqdm(coco_data['annotations']):
        image_id = annotation['image_id']
        image_info = img_map.get(image_id)
        if not image_info:
            continue

        file_name = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        bbox = annotation['bbox']
        original_category_id = annotation['category_id']
        yolo_class_id = category_id_to_yolo_id.get(original_category_id)
        if yolo_class_id is None:
            continue

        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width_norm = bbox[2] / img_width
        height_norm = bbox[3] / img_height

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        label_filename = Path(file_name).stem + '.txt'
        label_filepath = output_labels_dir / label_filename
        with open(label_filepath, 'a') as f: 
            f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        if file_name not in processed_images:
            src_image_path = image_dir / file_name
            dest_image_path = output_images_dir / file_name
            if src_image_path.exists() and not dest_image_path.exists():
                shutil.copy(src_image_path, dest_image_path)
            processed_images.add(file_name)

RAW_DATA_ROOT = Path('./data')
CONVERTED_DATA_ROOT = Path('./fisheye_yolo_dataset')
CLASS_NAMES = ['Bus', 'Bike', 'Car', 'Pedestrian', 'Truck']

if __name__ == "__main__":
    coco_to_yolo(
        coco_json_path=RAW_DATA_ROOT / 'train' / 'train.json',
        image_dir=RAW_DATA_ROOT / 'train' / 'images',
        output_root_dir=CONVERTED_DATA_ROOT / 'train',
        class_names=CLASS_NAMES
    )

    coco_to_yolo(
        coco_json_path=RAW_DATA_ROOT / 'test' / 'test.json',
        image_dir=RAW_DATA_ROOT / 'test' / 'images',
        output_root_dir=CONVERTED_DATA_ROOT / 'val',
        class_names=CLASS_NAMES
    )
