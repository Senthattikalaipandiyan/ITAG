"""
Data Augmentation Script for Indian Games Object Detection Dataset
Expands training split data (~1,000 images per class) using Albumentations.

Constraints:
- Zero Data Leakage: Only augments images listed in IndianGamesDataset/ImageSets/Main/train.txt
- Math-Safe Bounding Boxes: Parses float coordinates from Pascal VOC XMLs, transforms via albumentations A.BboxParams(format='pascal_voc'), and saves string coordinates.
"""

import os
import sys
import glob
import warnings
from pathlib import Path
import cv2
import numpy as np
from lxml import etree
import albumentations as A

# Suppress Albumentations ShiftScaleRotate warning in favor of explicit user requirement
warnings.filterwarnings('ignore', category=UserWarning)

# Directory configurations
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "IndianGamesDataset"
RAW_EXPORTS_DIR = BASE_DIR / "Raw_Exports"

TRAIN_TXT_PATH = DATASET_DIR / "ImageSets" / "Main" / "train.txt"
VAL_TXT_PATH = DATASET_DIR / "ImageSets" / "Main" / "val.txt"
IMAGES_DIR = DATASET_DIR / "JPEGImages"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"

AUG_DIR = DATASET_DIR / "Augmented"
AUG_IMAGES_DIR = AUG_DIR / "JPEGImages"
AUG_ANNOTATIONS_DIR = AUG_DIR / "Annotations"

NUM_VARIATIONS = 20


def ensure_dataset_structure():
    """Ensure IndianGamesDataset directory structure and splits are ready."""
    AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUG_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not TRAIN_TXT_PATH.exists() and RAW_EXPORTS_DIR.exists():
        print("[INIT] IndianGamesDataset structure not detected. Consolidating from Raw_Exports...")
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        TRAIN_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        train_list = []
        val_list = []
        for cat in os.listdir(RAW_EXPORTS_DIR):
            cat_dir = RAW_EXPORTS_DIR / cat
            if not cat_dir.is_dir():
                continue
            for xml in glob.glob(str(cat_dir / "Annotations" / "*.xml")):
                dest_xml = ANNOTATIONS_DIR / os.path.basename(xml)
                if not dest_xml.exists():
                    import shutil
                    shutil.copy2(xml, dest_xml)
            for jpg in glob.glob(str(cat_dir / "JPEGImages" / "*.jpg")):
                dest_jpg = IMAGES_DIR / os.path.basename(jpg)
                if not dest_jpg.exists():
                    import shutil
                    shutil.copy2(jpg, dest_jpg)
            for t in glob.glob(str(cat_dir / "ImageSets" / "Main" / "*_train.txt")):
                with open(t, "r") as f:
                    for line in f:
                        name = line.strip().split()[0] if line.strip() else None
                        if name:
                            train_list.append(name)
            for v in glob.glob(str(cat_dir / "ImageSets" / "Main" / "*_val.txt")):
                with open(v, "r") as f:
                    for line in f:
                        name = line.strip().split()[0] if line.strip() else None
                        if name:
                            val_list.append(name)
                            
        with open(TRAIN_TXT_PATH, "w") as f:
            for item in train_list:
                f.write(item + "\n")
        with open(VAL_TXT_PATH, "w") as f:
            for item in val_list:
                f.write(item + "\n")
        print(f"[INIT] Consolidate complete: {len(train_list)} train images, {len(val_list)} val images.")


def get_augmentation_pipeline():
    """
    Constructs the Albumentations composition with required transforms:
    - HorizontalFlip
    - VerticalFlip
    - RandomBrightnessContrast
    - ShiftScaleRotate
    with Pascal VOC bounding box support and boundary clipping.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        )
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.1,
        clip=True
    ))


def parse_voc_xml(xml_path, img_width, img_height):
    """
    Parses Pascal VOC XML, safely handling floating-point coordinates.
    Returns:
        bboxes: List of [xmin, ymin, xmax, ymax] floats
        labels: List of class name strings
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    bboxes = []
    labels = []
    
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        name = name_elem.text.strip() if name_elem is not None and name_elem.text else "unknown"
        
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
            
        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (ValueError, TypeError, AttributeError):
            continue
            
        # Math-safe bounds sanitization
        xmin = max(0.0, min(xmin, float(img_width - 1)))
        ymin = max(0.0, min(ymin, float(img_height - 1)))
        xmax = max(xmin + 1.0, min(xmax, float(img_width)))
        ymax = max(ymin + 1.0, min(ymax, float(img_height)))
        
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
        
    return bboxes, labels


def save_voc_xml(output_xml_path, folder_name, filename, image_path, width, height, depth, bboxes, class_labels):
    """
    Writes a standard Pascal VOC XML with casted string coordinates.
    """
    root = etree.Element("annotation", verified="yes")
    etree.SubElement(root, "folder").text = folder_name
    etree.SubElement(root, "filename").text = filename
    etree.SubElement(root, "path").text = str(image_path)
    
    source = etree.SubElement(root, "source")
    etree.SubElement(source, "database").text = "Unknown"
    
    size = etree.SubElement(root, "size")
    etree.SubElement(size, "width").text = str(width)
    etree.SubElement(size, "height").text = str(height)
    etree.SubElement(size, "depth").text = str(depth)
    
    etree.SubElement(root, "segmented").text = "0"
    
    for box, label in zip(bboxes, class_labels):
        xmin, ymin, xmax, ymax = box
        obj = etree.SubElement(root, "object")
        etree.SubElement(obj, "name").text = str(label)
        etree.SubElement(obj, "pose").text = "Unspecified"
        etree.SubElement(obj, "truncated").text = "0"
        etree.SubElement(obj, "difficult").text = "0"
        
        bndbox = etree.SubElement(obj, "bndbox")
        # Safely cast float coordinates back to strings with 3 decimal precision
        etree.SubElement(bndbox, "xmin").text = f"{float(xmin):.3f}"
        etree.SubElement(bndbox, "ymin").text = f"{float(ymin):.3f}"
        etree.SubElement(bndbox, "xmax").text = f"{float(xmax):.3f}"
        etree.SubElement(bndbox, "ymax").text = f"{float(ymax):.3f}"
        
    tree = etree.ElementTree(root)
    tree.write(str(output_xml_path), pretty_print=True, xml_declaration=False, encoding="utf-8")


def augment_training_data():
    """Main execution function for augmenting dataset."""
    ensure_dataset_structure()
    
    if not TRAIN_TXT_PATH.exists():
        print(f"[ERROR] train.txt not found at {TRAIN_TXT_PATH}")
        sys.exit(1)
        
    # Strictly read train.txt for Zero Data Leakage
    with open(TRAIN_TXT_PATH, "r") as f:
        train_image_ids = [line.strip().split()[0] for line in f if line.strip()]
        
    # Ensure uniqueness
    train_image_ids = sorted(list(set(train_image_ids)))
    total_train = len(train_image_ids)
    print(f"[INFO] Loaded {total_train} unique training images from train.txt.")
    print(f"[INFO] Generating {NUM_VARIATIONS} variations per training image (Expected: {total_train * NUM_VARIATIONS} images).")
    
    transform = get_augmentation_pipeline()
    total_generated = 0
    skipped_images = 0
    
    for idx, img_id in enumerate(train_image_ids, 1):
        jpg_path = IMAGES_DIR / f"{img_id}.jpg"
        xml_path = ANNOTATIONS_DIR / f"{img_id}.xml"
        
        if not jpg_path.exists() or not xml_path.exists():
            print(f"[WARNING] Missing pair for {img_id}: jpg={jpg_path.exists()}, xml={xml_path.exists()}. Skipping.")
            skipped_images += 1
            continue
            
        # Read image
        image = cv2.imread(str(jpg_path))
        if image is None:
            print(f"[WARNING] Failed to load image {jpg_path}. Skipping.")
            skipped_images += 1
            continue
            
        h, w, c = image.shape
        bboxes, labels = parse_voc_xml(xml_path, w, h)
        
        if not bboxes:
            print(f"[WARNING] No valid bounding boxes in {xml_path}. Skipping.")
            skipped_images += 1
            continue
            
        for var_idx in range(1, NUM_VARIATIONS + 1):
            aug_base_name = f"{img_id}_aug_{var_idx:02d}"
            aug_jpg_path = AUG_IMAGES_DIR / f"{aug_base_name}.jpg"
            aug_xml_path = AUG_ANNOTATIONS_DIR / f"{aug_base_name}.xml"
            
            # Apply transformation with retry to ensure bounding boxes are retained
            aug_boxes = []
            aug_labels = []
            aug_img = None
            
            for attempt in range(10):
                transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                if len(transformed['bboxes']) > 0:
                    aug_img = transformed['image']
                    aug_boxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    break
                    
            if aug_img is None or len(aug_boxes) == 0:
                # Fallback to mild brightness adjustment if spatial transforms clipped all boxes
                fallback_transform = A.Compose([
                    A.RandomBrightnessContrast(p=1.0)
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
                transformed = fallback_transform(image=image, bboxes=bboxes, class_labels=labels)
                aug_img = transformed['image']
                aug_boxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
            # Save augmented JPEG
            cv2.imwrite(str(aug_jpg_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            
            # Save recalculated Pascal VOC XML
            save_voc_xml(
                output_xml_path=aug_xml_path,
                folder_name="Augmented",
                filename=f"{aug_base_name}.jpg",
                image_path=str(aug_jpg_path),
                width=w,
                height=h,
                depth=c,
                bboxes=aug_boxes,
                class_labels=aug_labels
            )
            total_generated += 1
            
        if idx % 10 == 0 or idx == total_train:
            percent = (idx / total_train) * 100
            print(f"[{idx:3d}/{total_train}] Progress: {percent:5.1f}% | Generated: {total_generated} variations")
            
    print("\n" + "="*60)
    print(f"[SUCCESS] Data augmentation process completed!")
    print(f"Original training images processed : {total_train - skipped_images}")
    print(f"Total augmented images created     : {total_generated}")
    print(f"Output JPEGImages directory        : {AUG_IMAGES_DIR}")
    print(f"Output Annotations directory       : {AUG_ANNOTATIONS_DIR}")
    print("="*60)


if __name__ == "__main__":
    augment_training_data()
