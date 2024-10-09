import xml.etree.ElementTree as ET
import tensorflow as tf

# Define the XML file directory
xml_dir = '\\sort_py\\XML_Files'

# Define the output TFRecord file
output_file = 'sort_py\\train.record'

# Create a TFRecord writer
writer = tf.io.TFRecordWriter(output_file)

# Loop through each XML file in the directory
for file in os.listdir(xml_dir):
    if file.endswith('.xml'):
        # Parse the XML file
        tree = ET.parse(os.path.join(xml_dir, file))
        root = tree.getroot()

        # Extract the image file path, bounding box coordinates, and class labels
        image_file_path = root.find('filename').text
        bounding_boxes = []
        class_labels = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            class_label = obj.find('name').text
            bounding_boxes.append([x1, y1, x2, y2])
            class_labels.append(class_label)

        # Create a TFRecord example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file_path.encode('utf-8')])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[x1 / 224.0 for x1, _, _, _ in bounding_boxes])),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[y1 / 224.0 for _, y1, _, _ in bounding_boxes])),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[x2 / 224.0 for _, _, x2, _ in bounding_boxes])),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[y2 / 224.0 for _, _, _, y2 in bounding_boxes])),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_labels]))
        }))

        # Write the TFRecord example to the output file
        writer.write(example.SerializeToString())

# Close the TFRecord writer
writer.close()