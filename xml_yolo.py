import os
import sys  # Import sys to use sys.exit()
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def xml_to_yolo_bbox(bbox, w, h):
    """Convert XML bounding box coordinates to YOLO format."""
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def xml_to_yolo(xml_file, image_dir):
    """Convert XML annotation to YOLO format and return image path and YOLO label lines."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    img_path = os.path.join(image_dir, filename)
    
    if not os.path.exists(img_path):
        print(f"Warning: Image file {filename} not found in {image_dir}")
        return None, None
    
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    yolo_lines = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        bbox = [float(bbox.find(x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax']]
        yolo_bbox = xml_to_yolo_bbox(bbox, img_width, img_height)
        yolo_line = f"0 {' '.join([str(x) for x in yolo_bbox])}"  # Assuming 'mouse' is class 0
        yolo_lines.append(yolo_line)
    
    return img_path, yolo_lines

# Set up directories with raw strings to handle backslashes
xml_dir = r'C:\Varun_work\sort_py\XML_Files'  # Directory containing XML files
image_dir = r'C:\Varun_work\sort_py\JPG_Files'  # Directory containing image files
output_dir = r'mouse_dataset'
os.makedirs(output_dir, exist_ok=True)

# Create the 'labels' directory inside 'output_dir'
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Process XML files and collect image paths
image_files = []
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        img_path, yolo_lines = xml_to_yolo(os.path.join(xml_dir, xml_file), image_dir)
        if img_path and yolo_lines:
            image_files.append(img_path)
            # Write YOLO format labels
            txt_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            # Create the 'labels' directory if it doesn't exist
            with open(os.path.join(output_dir, 'labels', txt_filename), 'w') as f:
                f.write('\n'.join(yolo_lines))

print(f"Number of valid image files found: {len(image_files)}")

# Exit if no valid image files found
if len(image_files) == 0:
    print("No valid image files found. Please check your XML and image directories.")
    sys.exit()  # Use sys.exit() to exit the script

# Split dataset into training and validation sets
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Create directory structure for images and labels
for dir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(output_dir, dir), exist_ok=True)

# Move files to appropriate directories
for img in train_images:
    # Copy images to train directory
    shutil.copy(img, os.path.join(output_dir, 'images/train'))
    # Move corresponding label files to train directory
    txt = os.path.join(output_dir, 'labels', os.path.splitext(os.path.basename(img))[0] + '.txt')
    if os.path.exists(txt):
        shutil.move(txt, os.path.join(output_dir, 'labels/train'))

for img in val_images:
    # Copy images to validation directory
    shutil.copy(img, os.path.join(output_dir, 'images/val'))
    # Move corresponding label files to validation directory
    txt = os.path.join(output_dir, 'labels', os.path.splitext(os.path.basename(img))[0] + '.txt')
    if os.path.exists(txt):
        shutil.move(txt, os.path.join(output_dir, 'labels/val'))

# Create dataset.yaml file for YOLO
yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: 1
names: ['mouse']
"""

with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

print("Dataset preparation complete!")
