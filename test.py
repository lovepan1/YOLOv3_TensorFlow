print("[INFO] yolov3 netwrk initualized")
print("[INFO] loss calc completed")
a = 'yolov3/yolov3_head/Conv_51/weights'
try:
    nb = int(a.split('/')[-2].split('_')[-1]) + 1
except:
    nb = 1
print(nb)

# layer_name = 'yolov3/yolov3_head/Conv_1/weights'
# last_layer_name = 'yolov3/yolov3_head/Conv_' + str(int(layer_name.split('/')[-2].split('_')[-1]) -1) + '/weights'
# print(last_layer_name)

net_type = 'mobilenet_v1'
if net_type == 'mobilenet_v1':
    print('ok')