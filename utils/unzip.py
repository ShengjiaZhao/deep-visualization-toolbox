import os
import subprocess
import tarfile

files = os.listdir('imagenet')
for file in files:
	print("Extracting " + file)
	synset = file.split('.')[0]
	tar = tarfile.open('imagenet/' + file)
	os.mkdir('/sdb/' + synset)
	tar.extractall(path='/sdb/' + synset)
	tar.close()
	
