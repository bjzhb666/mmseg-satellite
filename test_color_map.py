import numpy as np
def generate_colormap():
    # 初始化colormap列表
    colormap = [[0, 0, 0]]

    # HSV颜色空间转换为RGB
    def hsv_to_rgb(h, s, v):
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            return (c + m, x + m, m)
        elif h < 120:
            return (x + m, c + m, m)
        elif h < 180:
            return (m, c + m, x + m)
        elif h < 240:
            return (m, x + m, c + m)
        elif h < 300:
            return (x + m, m, c + m)
        else:
            return (c + m, m, x + m)

    # 生成colormap
    for i in range(1, 256):
        # 计算色相，这里使用线性分布简化示例，实际应用可能需要更复杂的映射以获得更好的视觉效果
        hue = (i / 255.) * 360  # 色相在0到360度之间变化
        saturation = 0.8  # 饱和度保持较高
        value = i / 255. * 50 + 55  # 明度随灰度值线性变化

        # 将HSV转换为RGB
        rgb = hsv_to_rgb(hue, saturation, value)

        # 四舍五入并转换为整数，以便直接用于像素值
        colormap.append([int(round(c * 255)) for c in rgb])

    return np.array(colormap)

# by GPT4
def generate_colormap(num_colors=500):
    # 初始化colormap列表
    colormap = [[0, 0, 0]]

    # HSV颜色空间转换为RGB
    def hsv_to_rgb(h, s, v):
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            return (c + m, x + m, m)
        elif h < 120:
            return (x + m, c + m, m)
        elif h < 180:
            return (m, c + m, x + m)
        elif h < 240:
            return (m, x + m, c + m)
        elif h < 300:
            return (x + m, m, c + m)
        else:
            return (c + m, m, x + m)

    # 生成colormap
    for i in range(1, num_colors):
        hue = (i / num_colors) * 360  # 色相在0到360度之间变化
        saturation = 0.8  # 饱和度保持较高
        value = i / num_colors * 0.2 + 0.6  # 明度保持在60%到80%之间变化

        rgb = hsv_to_rgb(hue, saturation, value)

        # 四舍五入并转换为整数，以便直接用于像素值
        colormap.append([int(round(c * 255)) for c in rgb])

    return np.array(colormap)

# 示例用法
colormap = generate_colormap(500)
# colormap = generate_colormap_ori()
print(colormap)
# 检查colormap中是否有重复的颜色
print(len(colormap) == len(np.unique(colormap, axis=0)))  # True
print(len(np.unique(colormap, axis=0)))  # 500