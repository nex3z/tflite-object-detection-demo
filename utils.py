import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

LABEL_ID_OFFSET = 1


def load_data(labeled_dir, classes_file_path):
    tensor_images = []
    tensor_classes_one_hot = []
    tensor_boxes = []

    classes_map = build_classes_map(classes_file_path)
    num_classes = len(classes_map)

    for image_file in tqdm(glob.glob(os.path.join(labeled_dir, '*.jpg'))):
        image = Image.open(image_file)
        image = np.array(image)
        tensor_images.append(tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0))

        label_file = os.path.splitext(image_file)[0] + '.xml'
        label = parse_label(label_file)
        example_labels = []
        example_boxes = []
        for obj in label['objects']:
            example_labels.append(classes_map[obj['name']] - LABEL_ID_OFFSET)
            example_boxes.append(np.array(
                [obj['ymin'] / label['height'],
                 obj['xmin'] / label['width'],
                 obj['ymax'] / label['height'],
                 obj['xmax'] / label['width']], dtype=np.float32)
            )
        tensor_classes_one_hot.append(tf.one_hot(example_labels, num_classes))
        tensor_boxes.append(tf.convert_to_tensor(np.array(example_boxes), dtype=tf.float32))

    return tensor_images, tensor_boxes, tensor_classes_one_hot, classes_map


def build_classes_map(classes_file_path):
    with open(classes_file_path, 'r', encoding='utf8') as f:
        classes_text = f.read()
    classes = classes_text.split()
    classes_map = dict([(classes[i], i + 1) for i in range(len(classes))])
    return classes_map


def parse_label(label_file):
    content = ET.parse(label_file)
    root = content.getroot()

    filename = root.find('filename').text
    path = root.find('path').text

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        objects.append({
            'name': name,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        })

    return {
        'filename': filename,
        'path': path,
        'width': width,
        'height': height,
        'depth': depth,
        'objects': objects,
    }
