import os
import sys
import glob
import numpy as np

############ assume wav_root/speaker/wav_name ############
## Target: generate (wav_path, speaker) files
if (len(sys.argv)-1 != 2):
  print ("Usage: $0 <wav_root> <save_path>")
  print ("e.g. $0 ../../ICASSP5/database/wav ./speaker.txt")
  exit(1)

wav_dir = sys.argv[1]
save_path = sys.argv[2]

output = open(save_path, 'w')
#for sub_wav_dir in glob.glob(wav_dir+'/*'):
#	speaker = os.path.split(sub_wav_dir)[-1]
##    print("speaker:" + speaker)
#	# if speaker is int, then convert to str
#	if speaker.isdigit(): speaker = "speaker%05d" %(int(speaker))
#	assert speaker.isdigit()==False
#	for wav_path in glob.glob(sub_wav_dir+'/*'):
#		wav_name = os.path.split(wav_path)[-1]
#		wave_name, wav_type = wav_name.split('.')
#		if wav_type in ['wav', 'sph', 'flac']: output.write("%s %s\n" %(wav_path, speaker))


for root, dirs, files in os.walk(wav_dir):
    for file in files:
        path = os.path.join(root, file)
        file_name = os.path.split(path)
#        print(f"os.path.split(path): '{file_name}'")
        
        speaker = os.path.basename(os.path.dirname(root))
#        print("speaker:" + speaker)
        relative_path = os.path.relpath(path)  # 获取相对路径
#        path_name = os.path.dirname(relative_path)  # 获取相对路径的父目录，即去掉文件名和后缀
#        可能需要
#        path_name = os.path.join(os.path.basename(os.path.dirname(path_name)), os.path.basename(os.path.dirname(relative_path)), os.path.splitext(os.path.basename(relative_path))[0])  # 合并相对路径的倒数第三层路径和去掉后缀的文件名
#        
        path_type = os.path.splitext(path)[1]  # 获取文件后缀
        
#        print(f"speaker: '{speaker}'")
#        print(f"path_name: '{path_name}'")
#        print(f"path_type: '{path_type}'")
        
        if path_type.lower() in ['.flac', '.wav', '.sph']: # is audio
                output.write("%s %s\n" %(relative_path, speaker))

output.close()


