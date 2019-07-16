import os
import numpy as np
import cv2
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box


img_dir = '/home/pcl/data/VOC2007/JPEGImagesTest/'
anno_dir = '/home/pcl/data/VOC2007/Annotations'
gpu = open('gpu_inference_result_20190708.txt')
a200 = open('A200_result.txt')
anchor_path = './data/yolo_anchors.txt'
class_name_path = "./data/my_data/dianli_class.names"

anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)
print(num_class)
color_table = get_color_table(num_class)


f = open('7436.txt')
lines = f.readlines()
select_file_list = []
for i in lines:
    # print(i)
    if i == '\n':
        continue
    file = i.replace('\n', '')
    # print(file)
    select_file_list.append(file)
# print(select_file_list)

gpu_lines = gpu.readlines()
print(len(gpu_lines))
a200 = open('A200_result.txt')
def get_gpu_results():
    for line in gpu_lines:
        gpu_bboxes = []
        gpu_labels = []
        gpu_confs = []
        if line != '\n':
            line = line.replace('(', '').replace(')', '').replace(',', ' ').replace('\n', '')
            gpu_result = line.split(' ')
            file_name = gpu_result[0]
            obj_num = gpu_result[1]
            print(file_name)
            if file_name in select_file_list:
                img = cv2.imread(img_dir + file_name)
                height, width = img.shape[:2]
                for obj in range(int(obj_num)):
                    prune_line = gpu_result[2:]
                    label = int(prune_line[6*obj + 0])
                    center_x = float(prune_line[6*obj + 1])
                    center_y = float(prune_line[6 * obj + 2])
                    w = float(prune_line[6 * obj + 3])
                    h = float(prune_line[6 * obj + 4])
                    conf = float(prune_line[6 * obj + 5])

                    x_min = (center_x - w / 2) * width
                    y_min = (center_y - w / 2) * height
                    x_max = (center_x + w / 2) * width
                    y_max = (center_y + w / 2) * height
                    print(center_x, center_y, w, h)
                    # print(x_min, y_min, x_max, y_max)
                    gpu_bboxes.append([x_min, y_min, x_max, y_max])
                    gpu_labels.append(label)
                    gpu_confs.append(conf)

                for i in range(len(gpu_bboxes)):
                    x0, y0, x1, y1 = gpu_bboxes[i]
                    print(gpu_bboxes[i])
                    # print(classes[label[i]])
                    plot_one_box(img, [x0, y0, x1, y1], label=classes[gpu_labels[i]], color=color_table[gpu_labels[i]])
                cv2.imwrite('./gpu_results/' + file_name, img)

a200_lines = a200.readlines()
print(len(a200_lines))
def get_a200_results():
    for line in a200_lines:
        a200_bboxes = []
        a200_labels = []
        a200_confs = []
        if line != '\n':
            line = line.replace('(', '').replace(')', '').replace(',', '').replace('\n', '')
            a200_result = line.split(' ')
            file_name = a200_result[0]
            obj_num = a200_result[1]
            print(file_name)
            if file_name in select_file_list:
                img = cv2.imread(img_dir + file_name)
                height, width = img.shape[:2]
                for obj in range(int(obj_num)):
                    prune_line = a200_result[2:]
                    label = int(prune_line[6*obj + 0])
                    center_x = float(prune_line[6*obj + 1])
                    center_y = float(prune_line[6 * obj + 2])
                    w = float(prune_line[6 * obj + 3])
                    h = float(prune_line[6 * obj + 4])
                    conf = float(prune_line[6 * obj + 5])

                    x_min = (center_x - w / 2) * width
                    y_min = (center_y - w / 2) * height
                    x_max = (center_x + w / 2) * width
                    y_max = (center_y + w / 2) * height
                    print(center_x, center_y, w, h)
                    # print(x_min, y_min, x_max, y_max)
                    a200_bboxes.append([x_min, y_min, x_max, y_max])
                    a200_labels.append(label)
                    a200_confs.append(conf)

                for i in range(len(a200_bboxes)):
                    x0, y0, x1, y1 = a200_bboxes[i]
                    print(a200_bboxes[i])
                    # print(classes[label[i]])
                    plot_one_box(img, [x0, y0, x1, y1], label=classes[a200_labels[i]], color=color_table[a200_labels[i]])
                cv2.imwrite('./a200_results/' + file_name, img)
get_gpu_results()
get_a200_results()