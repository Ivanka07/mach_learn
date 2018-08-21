import glob
import shutil
import sys

source_dir = sys.argv[1]
target_dir = sys.argv[2]
split_char = sys.argv[3]

assert len(source_dir) > 0
assert len(target_dir) > 0
assert len(split_char) > 0 

dirs = glob.glob(source_dir  + '/*')
for d in dirs:
    csv = glob.glob(d + '/*.csv')
    for c in csv:
        csv_splitted = c.split(split_char)
        target = target_dir + split_char  + csv_splitted[1]
        shutil.move(c, target)
