__author__ = 'shengjia'


import sys
import os
import shutil

database_path = '/home/ubuntu/sdg'
labelfile_path = '/home/ubuntu/sdg/names'
listfile_path = '/home/ubuntu/sdg/database_list'

# database_path = '/home/shengjia/deep-visualization-toolbox/input_images'
# database_dest = 'output_image'

if __name__ == '__main__':
    if not os.path.isdir(database_path):
        print("Database not found")
        exit(1)
    files = os.listdir(database_path)
    dirs = []
    for cur_file in files:
        if os.path.isdir(database_path + '/' + cur_file):
            dirs.append(cur_file)
    dirs.sort()

    labelfile = open(labelfile_path, 'w')
    for cur_dir in dirs:
        labelfile.write(cur_dir + '\n')
    labelfile.close()
    listfile = open(listfile_path, 'w')
    for index, cur_dir in zip(range(0, len(dirs)), dirs):
        print(str(index) + ": Processing " + cur_dir)
        files = os.listdir(database_path + '/' + cur_dir)
        for cur_file in files:
            if os.path.isfile(database_path + '/' + cur_dir + '/' + cur_file):
                listfile.write(cur_dir + "/" + cur_file + " " + str(index) + "\n")
    listfile.close()