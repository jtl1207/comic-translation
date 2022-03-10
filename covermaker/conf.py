"""
这里是配置文件
"""


class Box(object):
    """
    box 定义文本框 x,y,w,h 都使用图片宽高的比例值
    并且图片原点为左上角，文本框锚点也是左上角
    """

    def __init__(self, x, y, w, h):
        self.lt = (x, y)  # 左上角
        self.rb = (x + w, y + h)  # 右下角
        self.w, self.h = w, h  # 宽和高


class Section(object):
    box = Box(0, 0, 0, 0)
    font = '华康翩翩体简-粗体.otf'  # font 使用哪个字体文件。这些字体存放在 fonts 目录下
    color = '#000000'  # 文本颜色
    trim = False  # trim false 超出范围后自动调整文本大小并换行  true 显示为单行，如果超过了显示范围就截断文本
    dir = 'v'  # 文本的显示方向。v 表示横向，h 表示纵向
    align = 'ct'  # 文本对齐方式 lt 表示垂直左对齐，水平顶端上对齐 {l r t b c} 分别表示 {左 右 上 下 中}
    valign = align[0]
    halign = align[1]
    line_spacing_factor = 0.3  # 行高  0是默认 1是2倍行高 -1是0
    letter_spacing_factor = 0.2  # 字符间距
    degree = 0  # 旋转角度
