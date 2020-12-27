import os
import json
import glob
import tqdm
from argparse import ArgumentParser
import ipdb;pdb=ipdb.set_trace
import numpy as np
import random
import cv2
import shutil


obj_list = ['diatom', 'diatom_2']
categories_list = [{"id": 1, "name": "diatom", "supercategory": "None"},{"id": 2, "name": "diatom_2", "supercategory": "None"}]

def read_json(name):
    try:
        try:
            with open(name, 'rt') as f:
                data = eval(eval(f.read()))
        except:
            data = json.load(open(name))
    except:
        with open(name, 'rt') as f:
            data = eval(f.read())

    return data


def write_json(name, data):
    with open(name, 'wt') as f:
        f.write(json.dumps(data))


# 根据图像，找到对应的object的标注
def find_items(images, anns):
    lists = []
    for img in images:
        image_id = img['id']
        for ann in anns:
            if image_id == ann['image_id']:
                lists.append(ann)
    return lists


class COCOLabel:
    def __init__(self, annotations_path, display_flag=0):
        self.annotations_path = annotations_path
        self.images = []
        self.anns = []
        self.display = display_flag
        self.create_annotations()
    

    def create_annotations(self):

        #json_paths = glob.glob(self.annotations_path+'/*.json')
        json_list = [item for item in os.listdir(self.annotations_path) if '.json' in item]
        json_paths = [os.path.join(self.annotations_path, item) for item in json_list]
        print(json_list)
        #ipdb.set_trace()
    
        object_id = 0
        image_id = 0
        for i,json_name in enumerate(json_list):
            image_id = i
            image_name = json_name.split('.json')[0] + '.bmp'
            img = {"id": image_id, "file_name": image_name}
            self.images.append(img)

            # image_path = os.path.join(self.annotations_path, image_name)
            # image_array = cv2.imread(image_path)
            # cv2.imshow(image_path, image_array)
            # cv2.waitKey(0)

            json_path = os.path.join(self.annotations_path, json_name)
            print("json_path = ", json_path)
            data = read_json(json_path)


            for item in data["shapes"]:
                x1, y1 = item['points'][0]
                x3, y3 = item['points'][2]
                w, h = x3-x1, y3-y1
                bbox = [x1, y1, w, h]
                area = w*h
                category = item['label']
                category_id = 0
                for item_c in categories_list:
                    if category == item_c['name']:
                        category_id = item_c['id']
                        break
                ann = {"id": object_id, "image_id": image_id, "category_id": category_id, "iscrowd": 0, 'area': area, "bbox": bbox}
                print('ann = ', ann)
                self.anns.append(ann)
                object_id += 1

            #     object_img = image_array[y1:y3, x1:x3]
            #     cv2.imshow('test', object_img)
            #     cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #ipdb.set_trace()


    def split_train_val(self, val_ratio = 0.1, images_path = '1500x', project_path = 'diatom'):
        random.seed(30)
        random.shuffle(self.images)
        val_num = int(len(self.images)*val_ratio)

        val_imgs = self.images[:val_num]
        val_anns = find_items(val_imgs, self.anns)

        train_imgs = self.images[val_num:]
        train_anns = find_items(train_imgs, self.anns)

        train_path = os.path.join(project_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        val_path = os.path.join(project_path, 'val')
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        anns_path = os.path.join(project_path, 'annotations')
        if not os.path.exists(anns_path):
            os.makedirs(anns_path)

        for img in val_imgs:
            image_name = img['file_name']
            file_image = os.path.join(images_path, image_name)
            shutil.copy(file_image, val_path)
        for img in train_imgs:
            image_name = img['file_name']
            file_image = os.path.join(images_path, image_name)
            shutil.copy(file_image, train_path)

        val_anns_path = os.path.join(anns_path, 'instances_val.json')
        val_data = {"categories": categories_list,
                    "images": val_imgs,
                    "annotations": val_anns}
        write_json(val_anns_path, val_data)

        train_anns_path = os.path.join(anns_path, 'instances_train.json')
        train_data = {"categories": categories_list,
                      "images": train_imgs,
                      "annotations": train_anns}
        write_json(train_anns_path, train_data)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-jp", "--json_path", help="json路径或者目录", default="1500x")
    args = parser.parse_args()
    annotations_path = args.json_path
    coco_label = COCOLabel(annotations_path)
    coco_label.split_train_val(val_ratio = 0.1, images_path = '1500x', project_path = 'diatom')
