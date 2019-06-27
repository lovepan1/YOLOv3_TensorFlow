import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document
import math
import random
from skimage import exposure
from skimage.util import random_noise
import copy


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            if filespath.split('.')[-1] == 'jpg' or filespath.split('.')[-1] == 'png':
                filepath = os.path.join(root, filespath)
                extension = os.path.splitext(filepath)[1][1:]
                if needExtFilter and extension in ext:
                    allfiles.append(filepath)
                elif not needExtFilter:
                    allfiles.append(filepath)
    return allfiles


def im_rotate(im, angle, center=None, scale=1.0):
    # try:
    h, w = im.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        im_rot = cv2.warpAffine(im, M, (w, h))
        return im_rot
    # except:
    #     print("the pic is corrupt")
    #     return 0


def readXml(xmlfile):
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    sizelist = annotation.getElementsByTagName('size')  # [<DOM Element: filename at 0x381f788>]
    heights = sizelist[0].getElementsByTagName('height')
    height = int(heights[0].childNodes[0].data)
    widths = sizelist[0].getElementsByTagName('width')
    width = int(widths[0].childNodes[0].data)
    depths = sizelist[0].getElementsByTagName('depth')
    depth = int(depths[0].childNodes[0].data)
    objectlist = annotation.getElementsByTagName('object')
    bboxes = []
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        class_label = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')[0]
        x1_list = bndbox.getElementsByTagName('xmin')
        x1 = int(int(x1_list[0].childNodes[0].data))
        y1_list = bndbox.getElementsByTagName('ymin')
        y1 = int(int(y1_list[0].childNodes[0].data))
        x2_list = bndbox.getElementsByTagName('xmax')
        x2 = int(int(x2_list[0].childNodes[0].data))
        y2_list = bndbox.getElementsByTagName('ymax')
        y2 = int(int(y2_list[0].childNodes[0].data))
        # 这里我box的格式【xmin，ymin，xmax，ymax，classname】
        bbox = [x1, y1, x2, y2, class_label]
        bboxes.append(bbox)
    return bboxes, width, height, depth


def rotate_image(angles, angle_rad, imgs_path, anno_new_path, folderName):
    j = 0
    angle_num = len(angles)
    for img_path in imgs_path:
        im = cv2.imread(img_path)
        for i in range(angle_num):
            gt_new = []
            im_rot = im_rotate(im, angles[i])
            #            if im_rot.all() == 0:
            #                return 0
            # try:
            (H, W, D) = im_rot.shape
            file_name = os.path.basename(os.path.splitext(img_path)[0])  # 得到原图的名称
            # 保存旋转后图像
            cv2.imwrite(os.path.join(pro_dir, 'XuanZhuan%s_%s.jpg' % (angles[i], file_name)),
                        im_rot)  # 新的命名方式为XuanZHuan+角度+原图名称
            # 读取anno标签数据，返回相应的信息
            # anno = os.path.join(imgs_path, '%s.xml' % file_name)
            anno = img_path.replace('jpg', 'xml')
            [gts, w, h, d] = readXml(anno)
            # 计算旋转后gt框四点的坐标变换
            [xc, yc] = [int(w) / 2, int(h) / 2]
            for gt in gts:
                # 计算左上角点的变换
                x1 = (gt[0] - xc) * math.cos(angle_rad[i]) - (yc - gt[1]) * math.sin(angle_rad[i]) + xc
                if int(x1) <= 0: x1 = 1
                if int(x1) > w - 1: x1 = w - 1
                y1 = yc - (gt[0] - xc) * math.sin(angle_rad[i]) - (yc - gt[1]) * math.cos(angle_rad[i])
                if int(y1) <= 0: y1 = 1.0
                if int(y1) > h - 1: y1 = h - 1
                # 计算右上角点的变换
                x2 = (gt[2] - xc) * math.cos(angle_rad[i]) - (yc - gt[1]) * math.sin(angle_rad[i]) + xc
                if int(x2) <= 0: x2 = 1.0
                if int(x2) > w - 1: x2 = w - 1
                y2 = yc - (gt[2] - xc) * math.sin(angle_rad[i]) - (yc - gt[1]) * math.cos(angle_rad[i])
                if int(y2) <= 0: y2 = 1.0
                if int(y2) > h - 1: y2 = h - 1
                # 计算左下角点的变换
                x3 = (gt[0] - xc) * math.cos(angle_rad[i]) - (yc - gt[3]) * math.sin(angle_rad[i]) + xc
                if int(x3) <= 0: x3 = 1.0
                if int(x3) > w - 1: x3 = w - 1
                y3 = yc - (gt[0] - xc) * math.sin(angle_rad[i]) - (yc - gt[3]) * math.cos(angle_rad[i])
                if int(y3) <= 0: y3 = 1.0
                if int(y3) > h - 1: y3 = h - 1
                # 计算右下角点的变换
                x4 = (gt[2] - xc) * math.cos(angle_rad[i]) - (yc - gt[3]) * math.sin(angle_rad[i]) + xc
                if int(x4) <= 0: x4 = 1.0
                if int(x4) > w - 1: x4 = w - 1
                y4 = yc - (gt[2] - xc) * math.sin(angle_rad[i]) - (yc - gt[3]) * math.cos(angle_rad[i])
                if int(y4) <= 0: y4 = 1.0
                if int(y4) > h - 1: y4 = h - 1
                xmin = min(x1, x2, x3, x4)
                xmax = max(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                ymax = max(y1, y2, y3, y4)
                # 把因为旋转导致的特别小的 长线型的去掉
                # w_new = xmax-xmin+1
                # h_new = ymax-ymin+1
                # ratio1 = int(w_new)/h_new
                # ratio2 = int(h_new)/w_new
                # if(1.0/6.0<ratio1<6 and 1.0/6.0<ratio2<6 and w_new>9 and h_new>9):
                classname = str(gt[4])
                gt_new.append([xmin, ymin, xmax, ymax, classname])
                # 写出新的xml
                writeXml(anno_new_path, 'XuanZhuan%s_%s' % (angles[i], file_name), W, H, D, gt_new, folderName,
                         img_path)
                print('roate', img_path)
            # except:
            #     print('fail to read %s picture ' % file_name)


def changeLiangDu(imgs_path, anno_new_path, folderName):
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        file_name = os.path.basename(os.path.splitext(img_path)[0])
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        flag = float('%.2f' % flag)
        changeLightImg = exposure.adjust_gamma(img, flag)
        cv2.imwrite(os.path.join(pro_dir_liangdu, 'changeLight%s_%s.jpg' % (flag, file_name)), changeLightImg)
        # anno = os.path.join(anno_new_path, '%s.xml' % file_name)
        anno = img_path.replace('jpg', 'xml')
        [gts, w, h, d] = readXml(anno)
        writeXml(anno_new_path, 'changeLight%s_%s' % (flag, file_name), w, h, d, gts, folderName, img_path)
        print('change light', img_path)


def addNoise(imgs_path, anno_new_path, folderName):
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        file_name = os.path.basename(os.path.splitext(img_path)[0])
        addNoiseImg = random_noise(img, mode='gaussian', clip=True) * 255
        cv2.imwrite(os.path.join(pro_dir_noise, 'addNoise%s.jpg' % file_name), addNoiseImg)
        # anno = os.path.join(anno_new_path, '%s.xml' % file_name)
        anno = img_path.replace('jpg', 'xml')
        [gts, w, h, d] = readXml(anno)
        writeXml(anno_new_path, 'addNoise%s' % file_name, w, h, d, gts, folderName, img_path)
        print('add noise', img_path)


def filpImages(imgs_path, anno_new_path, folderName):
    for img_path in imgs_path:
        gt_new = []
        img = cv2.imread(img_path)
        flip_img = copy.deepcopy(img)
        file_name = os.path.basename(os.path.splitext(img_path)[0])
        # anno = os.path.join(anno_new_path, '%s.xml' % file_name)
        anno = img_path.replace('jpg', 'xml')
        [gts, w, h, d] = readXml(anno)
        if random.random() < 0.5:
            horizon = True
        else:
            horizon = False
        if horizon:
            flip_img = cv2.flip(flip_img, 1)
        else:
            flip_img = cv2.flip(flip_img, 0)
        cv2.imwrite(os.path.join(pro_dir_flip, 'flipImg%s.jpg' % file_name), flip_img)
        for gt in gts:
            if horizon:
                x1 = w - gt[0]
                x2 = gt[1]
                x3 = w - gt[2]
                x4 = gt[3]
            else:
                x1 = gt[0]
                x2 = h - gt[1]
                x3 = gt[2]
                x4 = h - gt[3]

            if x1 < x3 and x2 < x4:
                xmin = x1
                ymin = x2
                xmax = x3
                ymax = x4
            if x1 >= x3 and x2 < x4:
                xmin = x3
                ymin = x2
                xmax = x1
                ymax = x4
            if x1 < x3 and x2 >= x4:
                xmin = x1
                ymin = x4
                xmax = x3
                ymax = x2
            if x1 < x3 and x2 <= x4:
                xmin = x3
                ymin = x2
                xmax = x1
                ymax = x4
            classname = str(gt[4])
            gt_new.append([xmin, ymin, xmax, ymax, classname])
        writeXml(anno_new_path, 'flipImg%s' % file_name, w, h, d, gt_new, folderName, img_path)
        print('flip Img', img_path)


def writeXml(tmp, imgname, w, h, d, bboxes, folderName, path):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode(folderName)
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname + os.path.splitext(path)[-1])
    filename.appendChild(filename_txt)

    #    path = doc.createElement('path')
    #    annotation.appendChild(path)
    #    path_txt = doc.createTextNode(str(path))
    #    path.appendChild(path_txt)

    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("unknown")
    database.appendChild(database_txt)

    #    annotation_new = doc.createElement('annotation')
    #    source.appendChild(annotation_new)
    #    annotation_new_txt = doc.createTextNode("VOC2007")
    #    annotation_new.appendChild(annotation_new_txt)

    #    image = doc.createElement('image')
    #    source.appendChild(image)
    #    image_txt = doc.createTextNode("flickr")
    #    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    #    flickrid = doc.createElement('flickrid')
    #    owner.appendChild(flickrid)
    #    flickrid_txt = doc.createTextNode("NULL")
    #    flickrid.appendChild(flickrid_txt)
    #
    #    ow_name = doc.createElement('name')
    #    owner.appendChild(ow_name)
    #    ow_name_txt = doc.createTextNode("idannel")
    #    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[4]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(int(int(bbox[0]))))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(int(int(bbox[1]))))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(int(int(bbox[2]))))
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(int(int(bbox[3]))))
        ymax.appendChild(ymax_txt)

        # print(int(int(bbox[0])),int(int(bbox[1])),int(int(bbox[2])),int(int(bbox[3])))

    tempfile = tmp + "/%s.xml" % imgname
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return


if __name__ == '__main__':
    # voc路径
    root = '/home/pcl/data/yangOngDisplay/'
    new_root = '/home/pcl/data/new_data/'
    # img_dir = root + '/imags'
    # anno_path = root + '/Anno'
    folderName = root.split("/")[-1]
    imgs_path = GetFileFromThisRootDir(root)  # 返回每一张原图的路径
    print(imgs_path)

    # 存储新的anno位置
    # anno_new_path = root + '/NewAnnotations'
    # #    anno_new_path_liangdu = root + '/NewAnnotationsLiangDu'
    # #    anno_new_path_noise = root + '/NewAnnotationsNoise'
    # #    anno_new_path_flip = root + '/NewAnnotationsFlip'
    # anno_new_path_liangdu = root + '/NewAnnotations'
    # anno_new_path_noise = root + '/NewAnnotations'
    # anno_new_path_flip = root + '/NewAnnotations'
    # if not os.path.isdir(anno_new_path):
    #     os.makedirs(anno_new_path)
    # if not os.path.isdir(anno_new_path_liangdu):
    #     os.makedirs(anno_new_path_liangdu)
    # if not os.path.isdir(anno_new_path_noise):
    #     os.makedirs(anno_new_path_noise)
    # if not os.path.isdir(anno_new_path_flip):
    #     os.makedirs(anno_new_path_flip)
    #
    # # 存储旋转后图片保存的位置
    # #    pro_dir = root+'/train_translate_scale_rotate/'
    # #    pro_dir_liangdu = root+'/train_change_liangdu/'
    # #    pro_dir_noise = root + '/train_add_noise/'
    # #    pro_dir_flip = root+'/train_flip/'
    # pro_dir = root + '/train_translate_scale_rotate/'
    # pro_dir_liangdu = root + '/train_translate_scale_rotate/'
    # pro_dir_noise = root + '/train_translate_scale_rotate/'
    # pro_dir_flip = root + '/train_translate_scale_rotate/'
    # if not os.path.isdir(pro_dir):
    #     os.makedirs(pro_dir)
    # if not os.path.isdir(pro_dir_liangdu):
    #     os.makedirs(pro_dir_liangdu)
    # if not os.path.isdir(pro_dir_noise):
    #     os.makedirs(pro_dir_noise)
    # if not os.path.isdir(pro_dir_flip):
    #     os.makedirs(pro_dir_flip)
    #

    anno_new_path = new_root
    anno_new_path_liangdu = new_root
    anno_new_path_noise = new_root
    anno_new_path_flip = new_root

    # 存储旋转后图片保存的位置
    pro_dir = new_root
    pro_dir_liangdu = new_root
    pro_dir_noise = new_root
    pro_dir_flip = new_root


    # 旋转角的大小，正数表示逆时针旋转
    angles = [5, 90, 180, 270, 355]  # 角度im_rotate用到的是角度制
    angle_rad = [angle * math.pi / 180.0 for angle in angles]  # cos三角函数里要用到弧度制的  
    #
    # # 开始旋转
    rotate_image(angles, angle_rad, imgs_path, anno_new_path, folderName)
    changeLiangDu(imgs_path, anno_new_path_liangdu, folderName)
    #addNoise(imgs_path, anno_new_path_noise, folderName)
    filpImages(imgs_path, anno_new_path_flip, folderName)