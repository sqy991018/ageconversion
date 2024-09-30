#import os
#import sys
#import glob
#
#if (len(sys.argv)-1 != 2):
#  print ("Usage: $0 <path-to-data> <output-dir>")
#  print ("e.g. $0 ./wav ./data")
#  exit(1)
#
#wav_path = sys.argv[1]
#out_dir = sys.argv[2]
#
#tmp_dir = out_dir + "/tmp"
#utt2spk_path = out_dir + "/utt2spk"
#wavscp_path = out_dir + "/wav.scp"
#
#if (os.system("mkdir -p %s" %(tmp_dir)) != 0):
#  print ("Error making directory %s" %(tmp_dir))
#
#
#output_1 = open(utt2spk_path, 'w')
#output_2 = open(wavscp_path, 'w')
#for path in glob.glob(wav_path + '/*'):
#	path_name, path_type = os.path.split(path)[-1].split('.')
#	if path_type in ['flac', 'wav', 'sph']: # is audio
#		speaker = path_name
#		utt2spk_temp = '%s %s\n' %(path_name, speaker)
#		if path_type == 'flac': wavscp_temp = '%s flac -c -d -s %s |\n' %(path_name, path)
#		if path_type == 'wav': wavscp_temp = '%s %s\n' %(path_name, path)
#		if path_type == 'sph': wavscp_temp = '%s sph2pipe -f wav -p -c 1 %s\n' %(path_name, path)
#	output_1.write(utt2spk_temp)
#	output_2.write(wavscp_temp)
#
#output_1.close()
#output_2.close()
#

import os
import sys
import glob

if (len(sys.argv)-1 != 2):
  print ("Usage: $0 <path-to-data> <output-dir>")
  print ("e.g. $0 ./wav ./data")
  exit(1)

wav_path = sys.argv[1]
out_dir = sys.argv[2]

tmp_dir = out_dir + "/tmp"
utt2spk_path = out_dir + "/utt2spk"
wavscp_path = out_dir + "/wav.scp"

if (os.system("mkdir -p %s" %(tmp_dir)) != 0):
  print ("Error making directory %s" %(tmp_dir))


output_1 = open(utt2spk_path, 'w')
output_2 = open(wavscp_path, 'w')

for root, dirs, files in os.walk(wav_path):
    for file in files:
        path = os.path.join(root, file)
        file_name = os.path.split(path)
#        print(f"os.path.split(path): '{file_name}'")
        
        speaker = os.path.basename(os.path.dirname(root))
        relative_path = os.path.relpath(path, start=wav_path)  # 获取相对路径
        path_name = os.path.dirname(relative_path)  # 获取相对路径的父目录，即去掉文件名和后缀
        path_name = os.path.join(os.path.basename(os.path.dirname(path_name)), os.path.basename(os.path.dirname(relative_path)), os.path.splitext(os.path.basename(relative_path))[0])  # 合并相对路径的倒数第三层路径和去掉后缀的文件名
        
        path_type = os.path.splitext(path)[1]  # 获取文件后缀
        
#        print(f"speaker: '{speaker}'")
#        print(f"path_name: '{path_name}'")
#        print(f"path_type: '{path_type}'")
        
        if path_type.lower() in ['.flac', '.wav', '.sph']: # is audio
            utt2spk_temp = '%s %s\n' %(path_name, speaker)
            if path_type.lower() == '.flac':
                wavscp_temp = '%s flac -c -d -s %s |\n' %(path_name, path)
            elif path_type.lower() == '.wav':
                wavscp_temp = '%s %s\n' %(path_name, path)
            elif path_type.lower() == '.sph':
                wavscp_temp = '%s sph2pipe -f wav -p -c 1 %s\n' %(path_name, path)
            output_1.write(utt2spk_temp)
            output_2.write(wavscp_temp)




output_1.close()
output_2.close()

