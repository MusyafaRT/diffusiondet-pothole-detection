import os
import json
import cv2
import numpy as np
import albumentations as A
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import hashlib
from PIL import Image
import threading

class DiffusionDetPreprocessor:
    def __init__(
        self,
        input_images_dir: str,
        input_annotations_dir: str,
        output_base_dir: str,
        train_ratio: float = 0.8,  # Adjusted for 665 images
        num_workers: int = 4,
        verify_images: bool = True,
        cache_enabled: bool = True,
        min_bbox_size: int = 32,  # Minimum bbox size for DiffusionDet
        max_instances_per_image: int = 100  # DiffusionDet recommended
    ):
        self.input_images_dir = input_images_dir
        self.input_annotations_dir = input_annotations_dir
        self.output_base_dir = output_base_dir
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.verify_images = verify_images
        self.cache_enabled = cache_enabled
        self.min_bbox_size = min_bbox_size
        self.max_instances_per_image = max_instances_per_image
        
        self.next_image_id = 1
        self._id_lock = threading.Lock()

        # Setup logging and directories
        self.setup_logging()
        self.setup_directories()
        
        # Initialize cache
        self.cache_dir = os.path.join(output_base_dir, '.cache')
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize statistics collector
        self.stats = {
            'processed_images': 0,
            'failed_images': 0,
            'augmented_images': 0,
            'skipped_images': 0,
            'truncated_instances': 0,
            'categories': set(),
            'image_sizes': [],
            'processing_times': []
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_base_dir, 'diffusion_processing.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary output directories for DiffusionDet"""
        self.dirs = {
            'train_images': os.path.join(self.output_base_dir, 'train'),  # DiffusionDet format
            'val_images': os.path.join(self.output_base_dir, 'val'),
            'train_annotations': os.path.join(self.output_base_dir, 'annotations'),
            'metadata': os.path.join(self.output_base_dir, 'metadata')
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)


    def get_augmentation_pipeline(self) -> A.Compose:
        """
        Augmentation pipeline optimized for pothole detection with focus on small objects.
        """
        return A.Compose([

            # Augmentasi flipping dan rotasi
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),

            # Augmentasi kecerahan dan kontras
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3),
                    contrast_limit=(-0.3, 0.3),
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.8),

            # Augmentasi noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.6),

            # Augmentasi blur dan efek lainnya
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=0.5),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.5),

            # Augmentasi simulasi kondisi lingkungan
            A.OneOf([
                A.RandomShadow(p=1.0),
                A.RandomRain(
                    slant_lower=-5, slant_upper=5, 
                    drop_length=10, 
                    p=0.5
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, 
                    fog_coef_upper=0.3, 
                    p=0.5
                ),
            ], p=0.5),

            # Augmentasi skala dan translasi
            A.Affine(
                scale=(0.9, 1.1), 
                translate_percent=(0.1, 0.2), 
                rotate=(-10, 10), 
                shear=(-5, 5), 
                p=0.8
            ),
        ], bbox_params=A.BboxParams(
            format='coco', 
            label_fields=['category_ids'], 
            min_visibility=0.5  # Minimum visibility threshold ditingkatkan
        ))

    def save_coco_annotations(self, split: str, processed_data: List[dict]):
        """Save annotations in COCO format for DiffusionDet"""
        coco_output = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': cat_id, 'name': f'class_{cat_id}'} 
                for cat_id in sorted(list(self.stats['categories']))
            ]
        }
        
        ann_id = 1
        for item in processed_data:
            # Ensure image_id is integer
            image_id = int(item['image_id'])
            
            # Add image info with actual dimensions
            coco_output['images'].append({
                'id': image_id,
                'file_name': item['file_name'],
                'height': item['height'],
                'width': item['width']
            })
            
            # Add annotations
            for bbox, category_id in zip(item['bboxes'], item['category_ids']):
                x, y, w, h = bbox
                coco_output['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(category_id),
                    'bbox': [float(coord) for coord in bbox],
                    'area': float(w * h),
                    'iscrowd': 0,
                    'segmentation': []
                })
                ann_id += 1
        
        # Save annotations
        output_file = os.path.join(
            self.dirs['train_annotations'],
            f'{split}.json'
        )
        with open(output_file, 'w') as f:
            json.dump(coco_output, f)



    def process_single_image(self, image_info: dict, annotations: List[dict]) -> List[Optional[dict]]:
        """Process a single image for DiffusionDet with multiple augmentations"""
        image_path = os.path.join(self.input_images_dir, image_info['file_name'])
        results = []
        
        if self.verify_images and not os.path.exists(image_path):
            self.logger.warning(f"Image not found: {image_path}")
            self.stats['failed_images'] += 1
            return results

        try:
            # Read image with PIL for better color handling
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Extract and filter boxes
            bboxes = [ann['bbox'] for ann in annotations]
            category_ids = [ann['category_id'] for ann in annotations]
            
            
            # Create multiple augmented versions
            transform = self.get_augmentation_pipeline()
            num_augmentations = 3

            # Original version
            original_filename = os.path.splitext(image_info['file_name'])[0]
            original_id = self.get_next_image_id()
            
            # Add original image with its dimensions
            height, width = image.shape[:2]
            results.append({
                'image': image,
                'bboxes': bboxes,
                'category_ids': category_ids,
                'image_id': original_id,
                'file_name': f"{original_filename}_orig.jpg",
                'height': height,
                'width': width
            })
            
            # Create augmented versions
            for aug_idx in range(num_augmentations):
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids
                )
                
                # Get next unique ID for augmented version
                aug_id = self.get_next_image_id()
                
                # Store augmented version with its dimensions
                aug_height, aug_width = augmented['image'].shape[:2]
                results.append({
                    'image': augmented['image'],
                    'bboxes': augmented['bboxes'],
                    'category_ids': augmented['category_ids'],
                    'image_id': aug_id,
                    'file_name': f"{original_filename}_aug{aug_idx+1}.jpg",
                    'height': aug_height,
                    'width': aug_width
                })
                
                self.stats['augmented_images'] += 1
            
            # Update statistics
            self.stats['processed_images'] += 1
            self.stats['categories'].update(category_ids)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            self.stats['failed_images'] += 1
            return results

    def save_processed_data(self, split: str, processed_data: List[dict]):
        """Save processed images and their annotations"""
        image_dir = self.dirs['train_images'] if split == 'train' else self.dirs['val_images']

        for item in tqdm(processed_data, desc=f"Saving {split} images"):
            output_path = os.path.join(image_dir, item['file_name'])
            
            # Convert to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(item['image'], cv2.COLOR_RGB2BGR)
            
            # Save image with high quality
            cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if self.verify_images and not os.path.exists(output_path):
                self.logger.warning(f"Failed to save image: {output_path}")


    def process_dataset(self):
        """Main processing pipeline for DiffusionDet"""
        self.logger.info("Starting DiffusionDet preprocessing pipeline...")
        
        self.next_image_id = 1

        # Load annotations
        self.logger.info("Loading annotations...")
        with open(os.path.join(self.input_annotations_dir, 'annotations.json'), 'r') as f:
            coco_data = json.load(f)
        
        # Group annotations by image
        ann_by_image = {}
        for ann in coco_data['annotations']:
            if ann['image_id'] not in ann_by_image:
                ann_by_image[ann['image_id']] = []
            ann_by_image[ann['image_id']].append(ann)
        
        # Process images in parallel
        self.logger.info("Processing images...")
        all_processed_data = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_info in coco_data['images']:
                future = executor.submit(
                    self.process_single_image,
                    img_info,
                    ann_by_image.get(img_info['id'], [])
                )
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing images"):
                results = future.result()  # Now returns list of results
                all_processed_data.extend(results)
        
        # Sort processed data by filename (this will ensure correct order)
        all_processed_data.sort(key=lambda x: x['file_name'])
        
        # Split dataset
        self.logger.info("Splitting dataset...")
        np.random.shuffle(all_processed_data)
        num_train = int(len(all_processed_data) * self.train_ratio)
        train_data = all_processed_data[:num_train]
        val_data = all_processed_data[num_train:]
        
        # Save images and annotations
        self.logger.info("Saving processed data...")
        self.save_processed_data('train', train_data)
        self.save_processed_data('val', val_data)
        
        # Save annotations in COCO format
        self.logger.info("Saving annotations...")
        self.save_coco_annotations('train', train_data)
        self.save_coco_annotations('val', val_data)
        
        # Save metadata
        self.save_metadata()
        
        self.logger.info(f"""Pipeline completed!
            Total images processed: {self.stats['processed_images']}
            Total augmentations created: {self.stats['augmented_images']}
            Training images: {len(train_data)}
            Validation images: {len(val_data)}
        """)
        
        return self.stats

    def save_metadata(self):
        """Save processing metadata and statistics"""
        metadata = {
            'dataset_stats': {
                'total_images': self.stats['augmented_images'],
                'train_images': int(self.stats['augmented_images'] * self.train_ratio),
                'val_images': int(self.stats['augmented_images'] * (1 - self.train_ratio)),
                'failed_images': self.stats['failed_images'],
                'truncated_instances': self.stats['truncated_instances'],
                'categories': list(self.stats['categories'])
            },
            'preprocessing_config': {
                'train_ratio': self.train_ratio,
                'min_bbox_size': self.min_bbox_size,
                'max_instances_per_image': self.max_instances_per_image
            }
        }
        
        with open(os.path.join(self.dirs['metadata'], 'preprocessing_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def get_next_image_id(self) -> int:
        with self._id_lock:
            current_id = self.next_image_id
            self.next_image_id += 1
            return current_id

if __name__ == '__main__':
    import os
    from pathlib import Path
    import time
    
    # Setup paths
    base_dir = Path(os.getcwd())
    input_images_dir = base_dir / "datasets" / "images"
    input_annotations_dir = base_dir / "datasets" 
    output_base_dir = base_dir / "preprocessed_dataset"
    
    # Verify input paths exist
    if not input_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {input_images_dir}")
    if not (input_annotations_dir / "annotations.json").exists():
        raise FileNotFoundError(f"Annotations file not found: {input_annotations_dir}/annotations.json")
    
    # Create output directory if it doesn't exist
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing dataset with following configuration:")
    print(f"Images directory: {input_images_dir}")
    print(f"Annotations directory: {input_annotations_dir}")
    print(f"Output directory: {output_base_dir}")
    
    try:
        start_time = time.time()
        
        preprocessor = DiffusionDetPreprocessor(
            input_images_dir=str(input_images_dir),
            input_annotations_dir=str(input_annotations_dir),
            output_base_dir=str(output_base_dir),
            train_ratio=0.8, 
            num_workers=4,
            verify_images=True
        )
        
        # Process dataset and get stats
        stats = preprocessor.process_dataset()
        
        # Print processing summary
        processing_time = time.time() - start_time
        print("\nProcessing completed successfully!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print("\nDataset statistics:")
        print(f"- Processed images: {stats['processed_images']}")
        print(f"- Failed images: {stats['failed_images']}")
        print(f"- Truncated instances: {stats['truncated_instances']}")
        print(f"- Number of categories: {len(stats['categories'])}")
        print(f"\nOutput saved to: {output_base_dir}")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise