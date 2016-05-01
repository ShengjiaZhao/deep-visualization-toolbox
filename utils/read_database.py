__author__ = 'shengjia'


import sys
import os
import shutil

database_path = '/home/ubuntu/sdg'
database_dest = '/home/ubuntu/sdf/images'
labelfile_path = '/home/ubuntu/sdf/names'
listfile_path = '/home/ubuntu/sdf/database_list'

# database_path = '/home/shengjia/deep-visualization-toolbox/input_images'
# database_dest = 'output_image'

if __name__ == '__main__':
    if not os.path.isdir(database_path):
        print("Database not found")
        exit(1)
    if not os.path.isdir(database_dest):
        os.mkdir(database_dest)
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
        print("Processing " + cur_dir)
        files = os.listdir(database_path + '/' + cur_dir)
        for cur_file in files:
            if os.path.isfile(database_path + '/' + cur_dir + '/' + cur_file):
                listfile.write(cur_file + " " + str(index) + "\n")
                shutil.copy(database_path + '/' + cur_dir + '/' + cur_file, database_dest)
    listfile.close()