import os
import shutil

src_dir = 'nusc1024'
dst_dir = 'nusc1024seg'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for i, filename in enumerate(os.listdir(src_dir)):
    if not (filename.endswith('line_num.png') or filename.endswith('line_type.png')):
        shutil.copy(os.path.join(src_dir, filename), dst_dir)

print(f'copy {len(os.listdir(dst_dir))} images')