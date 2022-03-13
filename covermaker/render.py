import os

import cv2
import numpy
from PIL import ImageDraw, Image, ImageOps, ImageFont
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
        if len(text.splitlines()) > 1:
            if section.dir == 'h':
                font_size = int(section.box.h / len(text.splitlines()) / (1+section.line_spacing_factor))
                ypos= int(font_size * section.line_spacing_factor + section.box.lt[1])
                xpos= int(section.box.w / 2 + section.box.lt[0])
                font_file = os.path.join(os.path.dirname(__file__), 'fonts', section.font)
                font = ImageFont.truetype(font_file, font_size)
                # 绘制
                for c in text.splitlines():
                    self._img_draw.text(xy=(xpos,ypos),
                                        text=c,
                                        fill=section.color,
                                        font=font,
                                        anchor='mt',
                                        align='center',
                                        stroke_width=section.stroke_width,
                                        stroke_fill=section.stroke_fill)
                    ypos += int(font_size * (1 + section.line_spacing_factor))
            else:
                # 对文本进行排版
                font_size = int(section.box.w / len(text.splitlines()) / (1 + section.line_spacing_factor))
                ypos = int(section.box.lt[1])
                xpos = int(section.box.rb[0] - font_size * (1 + section.line_spacing_factor) / 2)
                font_file = os.path.join(os.path.dirname(__file__), 'fonts', section.font)
                font = ImageFont.truetype(font_file, font_size)
                # 绘制
                for c in text.splitlines():
                    self._img_draw.text(xy=(xpos, ypos),
                                        text=c,
                                        direction='ttb',
                                        fill=section.color,
                                        font=font,
                                        anchor='mt',
                                        align='center',
                                        stroke_width=section.stroke_width,
                                        stroke_fill=section.stroke_fill)
                    xpos -= int(font_size * (1 + section.line_spacing_factor))
        else:
            # 未手动设置排版
            layout = layout_text(text, section)
            # 绘制
            for c, pos, section.degree, size in layout.iter_letters():
                self._img_draw.text(pos,
                                    c,
                                    fill=section.color,
                                    font=layout.font,
                                    stroke_width=section.stroke_width,
                                    stroke_fill=section.stroke_fill)

    def _draw_text_box(self, box):
        self._img_draw.rectangle((box.lt, box.rb), outline='#ff0000', width=2)

    def save(self, file_path):
        self._img.save(file_path)
