import os
f = open('./data/my_data/dianli_test.txt')
a = open('./data/my_data/dianli_test_right.txt', 'w')
list_file = f.readlines()
for line in list_file:
    if line.split('.')[-1] =='jpg\n':
        print(line)
        continue
    if line.split('.')[-1] =='JPG\n':
        print(line)
        continue
    if line.split('.')[-1] =='jpeg\n':
        print(line)
        continue
    else:
        a.write(line )
f.close()
a.close()