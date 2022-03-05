import cv2
import numpy
from PIL import ImageDraw, Image, ImageOps
from covermaker.layout import layout_text


class Render(object):
    def __init__(self, img_obj):
        self._img = Image.fromarray(cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB))
        self._img_draw = ImageDraw.Draw(self._img)

    def draw(self, text, text_conf_section, show_text_box=False):
        '''根据配置绘制文本

        Args:
            text (str): 待绘制的文本
            text_conf_section (Section object): 要绘制的文本的配置项
            show_text_box (bool, optional): 是否显示文本框的区域。文本框将绘制成红色
        '''
        # 绘制文本
        self._draw_text(text, text_conf_section)
        # 绘制文本框
        show_text_box and self._draw_text_box(text_conf_section.box)
        return cv2.cvtColor(numpy.asarray(self._img), cv2.COLOR_RGB2BGR)

    def _draw_text(self, text, section):
        if not text:
            return
        # 对文本进行排版
        layout = layout_text(text, section)
        # 开始绘制
        for c, pos, section.degree, size in layout.iter_letters():
            if section.degree == 0:
                self._img_draw.text(pos,
                                    c,
                                    fill=section.color,
                                    font=layout.font)
            else:
                # 旋转后文本的绘制。需要合并贴图
                img = Image.new('L', size)
                img_draw = ImageDraw.Draw(img)
                img_draw.text((0, 0), c, fill=255, font=layout.font)
                img_rotate = img.rotate(section.degree, expand=True)
                img_color = ImageOps.colorize(img_rotate, '#000000',
                                              section.color)
                self._img.paste(img_color, box=pos, mask=img_rotate)

    def _draw_text_box(self, box):
        self._img_draw.rectangle((box.lt, box.rb), outline='#ff0000', width=2)

    def save(self, file_path):
        self._img.save(file_path)
