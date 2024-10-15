rename_19_pics.py用于修改腾讯送过来奇怪的文件名

cut_satellite.py 用于把2048\*2048的图切成512\*512

generate_GT_direction_tag.py 用于生成GT的方向标签和tag的mask

generate_GT_multiclass_only.py 用于生成7个语义头的GT

train_val_test_gt.py 用于生成7个语义头+angle direction的训练集、验证集、测试集的GT

train_val_test_pic.py 用于生成训练集、验证集、测试集的图片

train_val_test_tag.py 用于生成训练集、验证集、测试集的tag（有npy文件所以单独列出）

get_black_images.py 用于获取训练集中全黑色的图片，生成txt文件供remove_black.py使用

remove_black.py 用于去除训练集中全黑色的图片

check_npy.sh 用于检查npy文件在切分后是否存在

# 使用流程

1. 生成7个语义头+angle direction的GT

   ```shell
   python generate_GT_multiclass_only.py
   ```
2. 生成tag和mask的GT

   ```shell
   python generate_GT_direction_tag.py
   ```
3. 生成训练集、验证集、测试集的图片

   ```shell
   python train_val_test_pic.py
   ```
4. 生成训练集、验证集、测试集的tag

   ```shell
   python train_val_test_tag.py
   ```
5. 生成训练集、验证集、测试集的其他语义GT

   ```shell
   python train_val_test_gt.py
   ```
6. 切分图片

   ```shell
   bash run_cut_satellite.sh
   ```
7. 检查npy文件是否存在

   ```shell
   bash check_npy.sh
   ```
8. 找到训练集测试集中全黑色的图片

   ```shell
   bash run_black_images.sh
   ```
9. 去除训练集中全黑色的图片（到此图片集构建完毕，后面是转换中文、mini数据集等等）

   ```shell
   bash remove_black_image.sh
   ```
10. 转换中英文标注脚本

    ```python
    python clear_json.py
    ```
11. 检查json是否存在中文

```python
python check_chinese.py
```

12. 创建mini数据集

```python
python create_mini.py
```

13. 创建mini的标注

```python
python create_gt_mini.py
```

14. 计算所有GT中每个像素（类别）出现的次数

```python
python calculate_pixel_level_nums.py
```

操作结束后，所有的图片在 `19picstrainvaltest-cut`文件夹下，所有的GT在 `19picsgt-cut`文件夹下，每个文件夹下有4个子文件夹：train、val、test、not_use_train，分别对应训练集、验证集、测试集、全黑训练集图。

Additional tools: 给定左上右下坐标统计区域内有多少各种级别的道路，在香港服务器中
