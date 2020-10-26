import glob
import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

import tensorflow as tf
from object_detection.utils import dataset_util

ITEM_TEMPLATE = """item {{
    name: '{name}',
    id: {id},
    display_name: '{display_name}'
}}
"""


def main():
    parser = build_parser()
    options = parser.parse_args()
    classes_map = build_classes_map(options.classes)
    print(f"classes_map = {classes_map}")

    writer = tf.io.TFRecordWriter(options.tf_record)

    for image_file in glob.glob(os.path.join(options.data_dir, '*.jpg')):
        label_file = os.path.splitext(image_file)[0] + '.xml'
        label = parse_label(label_file)
        tf_example = create_tf_example(image_file, label, classes_map)
        writer.write(tf_example.SerializeToString())

    writer.close()

    items = []
    for label_name, label_id in classes_map.items():
        item = ITEM_TEMPLATE.format(name=label_name, id=label_id, display_name=label_name)
        items.append(item)
    content = "\n".join(items)

    f = open(options.label_map, 'w', encoding='utf8')
    f.write(content)
    f.close()


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


def create_tf_example(image_file, label, classes_map):
    with tf.io.gfile.GFile(image_file, 'rb') as f:
        encoded_image_data = f.read()

    height = label['height']
    width = label['width']
    filename = label['filename'].encode('utf8')
    image_format = b'jpg'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for obj in label['objects']:
        xmins.append(obj['xmin'] / width)
        xmaxs.append(obj['xmax'] / width)
        ymins.append(obj['ymin'] / height)
        ymaxs.append(obj['ymax'] / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(classes_map[obj['name']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--labeled', dest='data_dir', default='./data/labeled')
    parser.add_argument('--classes', dest='classes', default='./data/classes.txt')
    parser.add_argument('--tf_record', dest='tf_record', default='./train.record')
    parser.add_argument('--label_map', dest='label_map', default='./label_map.pbtxt')
    return parser


if __name__ == '__main__':
    main()
