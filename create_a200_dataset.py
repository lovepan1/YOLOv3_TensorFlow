import xml.etree.ElementTree as ET
from os import getcwd
import os
sets=[('2007', 'train')]

classes = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "SuLiaoBu", "FengZheng", "Niao", "NiaoWo", "ShanHuo", "YanWu", "JianGeBang", "JueYuanZi", "FangZhenChui"]
root_dir = "/home/pcl/data/a200_dataset/JPEGImages"
anno_dir = "/home/pcl/data/a200_dataset/Annotations"
def convert_annotation(year, image_id, list_file, img_size):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    # anno_id =
    print(image_id)
    in_file = open(anno_dir+ '/%s.xml'% image_id.split('.')[0])
    tree=ET.parse(in_file)
    root = tree.getroot()
    for size in root.iter('size'):
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    # print(width)
    # print(height)
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        try:
            cls_id = classes.index(cls)
        except:
            continue
        xmlbox = obj.find('bndbox')


        b = (float(xmlbox.find('xmin').text)*img_size/width, float(xmlbox.find('ymin').text)*img_size/height, float(xmlbox.find('xmax').text)*img_size/width, float(xmlbox.find('ymax').text)*img_size/height)
        list_file.write(" " + str(cls_id) + " " + " ".join([str(a) for a in b]))

wd = getcwd()
import random
for year, image_set in sets:
    # image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
    img_size = random.sample(random_img_size, 1)[0][0]
    img_size = 416
    img_dir = root_dir + '/' + str(img_size)
    image_ids = os.listdir(img_dir)
    # print(image_ids)

    print("[INFO] the current img nums is %d" %len(image_ids))
    num = 0
    list_file = open('./data/my_data/a200_%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        print(image_id)
        num = num + 1
        if(num %1000 == 0):
            print("current deal img_num is %d"%num)
        list_file.write(str(num) + ' ' + img_dir + '/%s' %(image_id) )
        convert_annotation(year, image_id, list_file, img_size)
        num = num + 1
        list_file.write('\n')
    list_file.close()

