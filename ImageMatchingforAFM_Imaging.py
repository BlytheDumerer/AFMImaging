# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:13:50 2024

@author: Blythe Dumerer

This code implements template matching 

A cropped image is matched with a larger image to find similar components in the 
larger image. The resulting files are then broken down into a test/train split dataset for ML later
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import match_template
import imageio
import os
from sklearn.model_selection import train_test_split

class Template_Matching_CroppedImage_toLargerImage:
    def __init__(self, template_directory, image_directory, annotation_directory):
        self.template_directory = template_directory
        self.image_directory = image_directory
        self.annotation_directory = annotation_directory
        self.straight_templates = None
        self.curled_templates = None
        self.load_templates()
        
    """Calculates the Intersection over Union (IoU) of two bounding boxes, 
    which is a measure of how much the boxes overlap. It is used to filter out 
    overlapping regions."""
    
    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(xi2 - xi1, 0)
        inter_height = max(yi2 - yi1, 0)
        inter_area = inter_width * inter_height
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area
    
    """Filter out the overlapping boxes based on the threshold."""

    def filter_overlapping_boxes(self, boxes, threshold=0.1):
        filtered_boxes = []
        for box in boxes:
            if all(self.iou(box, filtered_box) < threshold for filtered_box in filtered_boxes):
                filtered_boxes.append(box)
        return filtered_boxes
    
    """The main template matching function and parameters that are needed to be defined."""

    def MultipleMatch(self, threshold=None):
        try:
            image = np.load(self.image_directory)
        except FileNotFoundError:
            print(f"File not found: {self.image_directory}")
            raise
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
    
        if image.ndim == 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image
    
        if gray_image.ndim != 2:
            print("Error: Grayscale image is not 2D")
            raise ValueError("Grayscale image is not 2D")
            
        def extract_template(yrange1, yrange2, xrange1, xrange2):
            try:
                return gray_image[yrange1:yrange2, xrange1:xrange2]
            except IndexError:
                print("Error: The subregion indices are out of bounds for the image")
                raise

        def convert_to_yolo_format(bbox, img_width, img_height, class_id):
            x, y, w, h = bbox
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            return f"{class_id} {x_center} {y_center} {width} {height}"

        def save_template_and_annotation(templates, template_type, class_id, img_width, img_height, i):
            os.makedirs(template_type, exist_ok=True)
            for j, template in enumerate(templates, start=1):
                template_filename = f'{template_type}_{j}'
                template_path = os.path.join(template_type, f'{template_filename}.png')
                annotation_path = os.path.join(template_type, f'annotations{i}_{template_filename}.txt')

                imageio.imwrite(template_path, template)

                annotation = convert_to_yolo_format((0, 0, template.shape[1], template.shape[0]), img_width, img_height, class_id)

                with open(annotation_path, 'w') as f:
                    f.write(annotation + '\n')
        
        save_template_and_annotation(self.straight_templates, 'straight', 0, gray_image.shape[1], gray_image.shape[0], i)
        save_template_and_annotation(self.curled_templates, 'curled', 1, gray_image.shape[1], gray_image.shape[0], i)
        
        """ Match the templates to the image but aim to not identify multiple sections of the same image
        (threshold criteria)"""
        
        def match_and_plot(templates, template_name, class_id):
            all_boxes = []
            for template in templates:
                if template.ndim != 2:
                    print(f"Error: Template ({template_name}) is not 2D")
                    raise ValueError(f"Template ({template_name}) is not 2D")
                result = match_template(gray_image, template)
                match_locations = np.where(result >= threshold)
                
                boxes = [(x, y, template.shape[1], template.shape[0]) for (y, x) in zip(*match_locations)]
                all_boxes.extend(boxes)
    
            filtered_boxes = self.filter_overlapping_boxes(all_boxes)
            
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            
            for i, template in enumerate(templates):
                axs[i // 2, i % 2].imshow(template, cmap=plt.cm.gray)
                axs[i // 2, i % 2].set_axis_off()
                axs[i // 2, i % 2].set_title(f'{template_name} Template {i+1}')
    
            axs[1, 0].imshow(gray_image, cmap=plt.cm.gray)
            axs[1, 0].set_axis_off()
            axs[1, 0].set_title('Image')
    
            for (x, y, w, h) in filtered_boxes:
                rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
                axs[1, 0].add_patch(rect)
                    
                matched_region = gray_image[y:y+h, x:x+w]
                plt.figure()
                plt.imshow(matched_region, cmap=plt.cm.gray)
                plt.title(f"Matched region at x={x}, y={y}")
                plt.axis('off')
                plt.close()
    
            result = match_template(gray_image, templates[0])
            match_locations = np.where(result >= threshold)
            axs[1, 1].imshow(result)
            axs[1, 1].set_axis_off()
            axs[1, 1].set_title('match_template\nResult')
            axs[1, 1].autoscale(False)
            axs[1, 1].plot(match_locations[1], match_locations[0], 'o', markeredgecolor='r', 
                           markerfacecolor='none', markersize=10)
                
            plt.show()
            
            return filtered_boxes
    
        straight_boxes = match_and_plot(self.straight_templates, 'Straight', class_id=0)
        curled_boxes = match_and_plot(self.curled_templates, 'Curled', class_id=1)
        
        all_boxes = straight_boxes + curled_boxes
        class_ids = [0] * len(straight_boxes) + [1] * len(curled_boxes)
    
        def save_yolo_annotations(bboxes, img_width, img_height, class_ids, i):
            annotations = []
            for bbox, class_id in zip(bboxes, class_ids):
                yolo_format = convert_to_yolo_format(bbox, img_width, img_height, class_id)
                annotations.append(yolo_format)
        
            output_annotations_file = os.path.join(self.annotation_directory, f'annotations{i}.txt')
            with open(output_annotations_file, 'w') as f:
                for annotation in annotations:
                    f.write(annotation + '\n')

        save_yolo_annotations(all_boxes, gray_image.shape[1], gray_image.shape[0], class_ids, i)

    def load_templates(self):
        try:
            image = np.load(os.path.join(self.template_directory, 'file18.npy'))
        except FileNotFoundError:
            print(f"File not found: {os.path.join(self.template_directory, 'file18.npy')}")
            raise
        except Exception as e:
            print(f"Error loading template image: {e}")
            raise
    
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
    
        if image.ndim == 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image
    
        if gray_image.ndim != 2:
            print("Error: Grayscale image is not 2D")
            raise ValueError("Grayscale image is not 2D")
            
        def extract_template(yrange1, yrange2, xrange1, xrange2):
            try:
                return gray_image[yrange1:yrange2, xrange1:xrange2]
            except IndexError:
                print("Error: The subregion indices are out of bounds for the image")
                raise

        self.straight_templates = [
            extract_template(125, 149, 535, 568),
            extract_template(87, 126, 884, 900)
        ]
        self.curled_templates = [
            extract_template(108, 130, 600, 650),
            extract_template(416, 442, 743, 771)
        ]
        
    """ Take the rows of the collected annotation data and shuffle them, then split them into
    testing and training files continually adding to the files through each annotation set until
    all of the annotation files created earlier have been divided into testing and training 
    datasets"""
    
    def split_annotations(self, split_files, train_path, test_path):
        train_annotations = []
        test_annotations = []
        
        for split_file in split_files:
            annotation_file = os.path.join(self.annotation_directory, f'annotations{split_file}.txt')
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                train_lines, test_lines = train_test_split(lines, test_size=0.60, train_size=0.40, shuffle=True)
                train_annotations.extend(train_lines)
                test_annotations.extend(test_lines)

        # Save train annotations
        with open(train_path, 'w') as train_file:
            for line in train_annotations:
                train_file.write(line)

        # Save test annotations
        with open(test_path, 'w') as test_file:
            for line in test_annotations:
                test_file.write(line)






# Instantiate the Template_Matching_CroppedImage_toLargerImage class
template_directory = r'C:\Users\Documents\OR\python_flatten'
image_directory_pattern = r'C:\Users\Documents\OR\python_flatten\file{}.npy'
split_files = [1,2,3,5,6,7,8,9,10,11,13,14,15,16,17] # select which files you want for validation and remove them from this list
train_path = r'C:\Users\Documents\OR\train_files.txt' #location where you want the train files
test_path = r'C:\Users\Documents\OR\test_files.txt' #location where you want the test files
annotation_directory = r'C:\Users\Documents\OR' #location of where you want the annotation files

# Split and save train/test files
matcher = Template_Matching_CroppedImage_toLargerImage(template_directory, image_directory_pattern.format(1), annotation_directory)
matcher.split_annotations(split_files, train_path, test_path)

# Read the split files
def read_split_files(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

train_files = read_split_files(train_path)
test_files = read_split_files(test_path)

# Run MultipleMatch for each file in the train and test sets
for i, annotation_file in enumerate(train_files, start=1):
    image_directory = image_directory_pattern.format(i)
    matcher = Template_Matching_CroppedImage_toLargerImage(template_directory, image_directory, annotation_directory)
    matcher.MultipleMatch(threshold=0.3)

for i, annotation_file in enumerate(test_files, start=1):
    image_directory = image_directory_pattern.format(i)
    matcher = Template_Matching_CroppedImage_toLargerImage(template_directory, image_directory, annotation_directory)
    matcher.MultipleMatch(threshold=0.3)
    
    """ A file not found error will appear at the end when the file is complete and there
     Are no more files to process"""