dataset_type = 'SatelliteInstanceDataset'
# data_root = 'data/satellite/'
data_root = 'data/satellite19/'
crop_size = (2048, 2048) # random crop size

dilate_kernel = 9 # 不能单独dilate，如果真的想后面的instance tag也需要跟着修改，这里暂定不改
test_dilate_kernel=3
train_pipeline = [
    dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
    dict(type='LoadInstanceAnnotations'),  # 第2个流程，对于当前图像，加载它的标注图像
    dict(type='Resize',  # 调整输入图像大小(resize)和其标注图像的数据增广流程
    scale=crop_size),  # 图像裁剪的大小
    dict(type='DilateGT', kernel_size=dilate_kernel),  # 对标注图像进行膨胀操作    
    dict(type='RandomFlip',  # 翻转图像和其标注图像的数据增广流程
        prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),  # 翻转图像的概率，0.5概率不翻转，0.5/3的概率执行每个翻转操作
    # dict(type='NormalizeLineCoordinate', img_shape=crop_size),  # 标准化线的坐标
    dict(type='PhotoMetricDistortion'),  # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程，和GT无关，不用改
    dict(type='PackInstanceSegInputs')  # 打包用于语义分割的输入数据
]

test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第1个流程，从文件路径里加载图像
    dict(type='Resize',  # 调整输入图像大小(resize)和其标注图像的数据增广流程
    scale=crop_size),  # 图像裁剪的大小
    # 在' Resize '之后添加标注图像
    # 不需要做调整图像大小(resize)的数据变换  
    dict(type='LoadInstanceAnnotations'),  # 加载数据集提供的语义分割标注
    dict(type='DilateGT', kernel_size=test_dilate_kernel),  # 对标注图像进行膨胀操作   
    dict(type='PackInstanceSegInputs')  # 打包用于语义分割的输入数据
]

train_dataloader = dict(  # 训练数据加载器(dataloader)的配置
    batch_size=2,  # 每一个GPU的batch size大小
    num_workers=2,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 训练时进行随机洗牌(shuffle)
    dataset=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='img_dir/small_test', seg_map_path='mask_tag/small_test'),  # 训练数据的前缀
        direction_path='angle_direction/small_test',
        color_path='color/small_test',
        line_type_path='line_type/small_test',
        line_num_path='num/small_test',
        attribute_path = 'attribute/small_test',
        ifbidirection_path = 'direction/small_test',
        ifboundary_path = 'boundary/small_test',
        pipeline=train_pipeline)) # 数据处理流程，它通过之前创建的train_pipeline传递。

val_dataloader = dict(
    batch_size=1,  # 每一个GPU的batch size大小
    num_workers=4,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时不进行随机洗牌(shuffle)
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='img_dir/small_test', seg_map_path='mask_tag/small_test'),  # 测试数据的前缀
        direction_path='angle_direction/small_test',
        color_path='color/small_test',
        line_type_path='line_type/small_test',
        line_num_path='num/small_test',
        attribute_path = 'attribute/small_test',
        ifbidirection_path = 'direction/small_test',
        ifboundary_path = 'boundary/small_test',
        pipeline=test_pipeline))  # 数据处理流程，它通过之前创建的test_pipeline传递。

test_dataloader = dict(
    batch_size=4,  # 每一个GPU的batch size大小
    num_workers=4,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时不进行随机洗牌(shuffle)
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='img_dir/small_test', seg_map_path='mask_tag/small_test'),  # 测试数据的前缀
        direction_path='angle_direction/small_test',
        color_path='color/small_test',
        line_type_path='line_type/small_test',
        line_num_path='num/small_test',
        attribute_path = 'attribute/small_test',
        ifbidirection_path = 'direction/small_test',
        ifboundary_path = 'boundary/small_test',
        pipeline=test_pipeline))  # 数据处理流程，它通过之前创建的test_pipeline传递。

# test_dataloader = dict(
#     batch_size=1,  # 每一个GPU的batch size大小
#     num_workers=4,  # 为每一个GPU预读取数据的进程个数
#     persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),  # 训练时不进行随机洗牌(shuffle)
#     dataset=dict(  # 测试数据集配置
#         type=dataset_type,  # 数据集类型，详见mmseg/datassets/
#         data_root=data_root,  # 数据集的根目录
#         data_prefix=dict(
#             img_path='img_dir/small_test2', seg_map_path='mask_tag/small_test2'),  # 测试数据的前缀
#         direction_path='angle_direction/small_test2',
#         color_path='color/small_test2',
#         line_type_path='line_type/small_test2',
#         line_num_path='num/small_test2',
#         attribute_path = 'attribute/small_test2',
#         ifbidirection_path = 'direction/small_test2',
#         ifboundary_path = 'boundary/small_test2',
#         pipeline=test_pipeline))  # 数据处理流程，它通过之前创建的test_pipeline传递。


# 精度评估方法，我们在这里使用 IoUMetric 进行评估
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice', 'mFscore'])
test_evaluator = val_evaluator