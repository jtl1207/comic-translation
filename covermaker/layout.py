'''
@Author: 张宇飞
@Description: 本文件的类和算法皆为排版服务。其核心思想是在一个固定大小的显示范围内找到一个能够
最大限度利用显示区域的字号，并让显示的内容符合基本的排版规则。
关于字号的匹配算法我的思路是这样的：

字号适配算法：
    1. 求最大最小区间。因为考虑到字符可能不等宽。所以下限设置为固定值 2
       最大值设置为单行存放所有字的情况
    3. 在 [minx, maxx] 范围内通过2分查找的算法：
        在能按“换行规则”都放入文本框内的时候，向更大的字号靠近
        在不能放入文本框内的时候，向更小的字号靠近

关于换行规则：
    1. 使用正则将字符串中间的连续单词或数字块找出来。这些词汇不可以在换行的时候被分割
    2. 因为行或列的长度固定，使用贪心算法，向一行添加分割好的英文词汇或者中文字符，使之能盛放最多数据
        插入第一个字符的时候需要判定是否符合排版规则，如果不符合，就要向前一行尾部借数据
    3. 新建一行。判定上一行结尾字符是否符合排版规则
        是 -> 跳转到过程 2
        否 -> 插入到当前行首。再次判定上一行结尾字符
@Email: zhangyufei49@163.com
'''

import os
import re
from collections import deque
from unicodedata import east_asian_width

from PIL import ImageFont

_DEBUG = True

_MIN_FONT_SIZE = 2


def _is_wide_char(uchar):
    w = east_asian_width(uchar)
    return w == 'W' or w == 'F'


def _get_font(font_name, font_size):
    cur_dir = os.path.dirname(__file__)
    font_file = os.path.join(cur_dir, 'fonts', font_name)
    return ImageFont.truetype(font_file, font_size)


# 用于提取单词和数字
_RE_EN_NUM_WORDS = re.compile(r"\w+|[!?#$%.,&|\"\'-_]+", re.ASCII)


def _splite_text_to_words(text):
    '''将文本切分成单词。非宽字符将是单个的字，英文和数字将是词组

    Returns:
        list : 例如输入 '豆瓣2020 hello' 将返回 ['豆', '瓣', '2020', ' ', 'hello']
    '''
    ascii_words_range = ((x.start(), x.end())
                         for x in _RE_EN_NUM_WORDS.finditer(text))
    i = 0
    ret = []
    for r in ascii_words_range:
        while i < r[0]:
            ret.append(text[i])
            i += 1
        ret.append(text[r[0]:r[1]])
        i = r[1]
    while i < len(text):
        ret.append(text[i])
        i += 1
    return ret


# 行首禁止出现的标点符号
_PUNCTUATION_BLOCK_SET = {
    ',',
    '.',
    ':',
    ';',
    '!',
    '?',
    ')',
    '}',
    ']',
    '\'',
    '，',
    '。',
    '：',
    '；',
    '！',
    '？',
    '）',
    '】',
    '、',
    '》',
    '…',
    '”',
    '’',
}


class Line(object):
    '''行。每行保存了很多词。'''

    def __init__(self, font, letter_spacing):
        self.words = deque()
        self._font = font
        self._letter_spacing = letter_spacing
        self._words_width = 0
        self._letter_count = 0

    def _update(self, word, sign):
        self._letter_count += sign * len(word)
        self._words_width += sign * self._font.getsize(word)[0]

    def append(self, word):
        self.words.append(word)
        self._update(word, 1)

    def append_left(self, word):
        self.words.appendleft(word)
        self._update(word, 1)

    def pop(self):
        word = self.words.pop()
        self._update(word, -1)
        return word

    def pop_left(self):
        word = self.words.popleft()
        self._update(word, -1)
        return word

    def get_display_width(self):
        '''返回当前行所有字在排版后的宽度。包含了字符间距'''
        ls = (self._letter_count - 1) * self._letter_spacing
        return int(ls + self._words_width)

    def __str__(self):
        return ''.join(self.words)


class Layout(object):
    '''排版后最终向外展示的类'''

    def __init__(self, lines, font, font_size, line_spacing, letter_spacing):
        self.lines = lines
        self.font = font
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing

        self._lines_start_pos = [[0, 0] for _ in range(len(lines))]
        self._lines_height = len(lines) * (font_size +
                                           line_spacing) - line_spacing
        self._dir = None

    def update(self, text_box, dir, valign, halign):
        # 执行一次重排版
        self._dir = dir
        if dir == 'h':
            self._update_h(text_box, valign, halign)
        else:
            self._update_v(text_box, valign, halign)

    def iter_letters(self):
        '''遍历所有的单个字，获取他们的排版信息
        可用于绘制。

        Yields:
            str, tuple, int, tuple: 单个字，字的左上角坐标(x, y) ，旋转角度，(宽度，高度)
        '''
        if not self._dir:
            return
        if self._dir == 'h':
            for i, line in enumerate(self.lines):
                pos = self._lines_start_pos[i]
                x = pos[0]
                for word in line.words:
                    for c in word:
                        fw = self.font.getsize(c)[0]
                        fh = self.font.getsize(c)[1]
                        yield c, (x, pos[1]), 0, (fw, fh)
                        x += fw + self.letter_spacing
        else:
            for i, line in enumerate(self.lines):
                pos = self._lines_start_pos[i]
                y = pos[1]
                for word in line.words:
                    for c in word:
                        is_wide = _is_wide_char(c)
                        degree = 0 if is_wide else -90
                        fw = self.font.getsize(c)[1]
                        fh = self.font.getsize(c)[0]
                        # 这里的宽高仍旧按照横向写，你需要根据角度自行计算
                        yield c, (pos[0], y), 0, (fw, fh)
                        y += fw + self.letter_spacing

    def _update_h(self, box, valign, halign):
        for i, line in enumerate(self.lines):
            # 求 x 坐标
            if valign == 'l':
                xoff = 0
            elif valign == 'r':
                xoff = box.w - line.get_display_width()
            else:
                xoff = (box.w - line.get_display_width()) / 2
            # 求 y 坐标
            yoff = i * (self.line_spacing + self.font_size)
            if halign == 'b':
                yoff += box.h - self._lines_height
            elif halign == 'c':
                yoff += (box.h - self._lines_height) / 2
            self._lines_start_pos[i][0] = int(box.lt[0]) + int(xoff)
            self._lines_start_pos[i][1] = int(box.lt[1]) + int(yoff)

    def _update_v(self, box, valign, halign):
        for i, line in enumerate(self.lines):
            # 求 x 坐标
            xoff = self.font_size + (self.font_size + self.line_spacing) * i
            if valign == 'l':
                xoff += (box.w - self._lines_height)
            elif valign == 'c':
                xoff += (box.w - self._lines_height) / 2
            # 求 y 坐标
            if halign == 't':
                yoff = 0
            elif halign == 'b':
                yoff = box.h - line.get_display_width()
            else:
                yoff = (box.h - line.get_display_width()) / 2
            self._lines_start_pos[i][0] = int(box.rb[0]) - int(xoff)
            self._lines_start_pos[i][1] = int(box.lt[1]) + int(yoff)


def _build_lines(text, font, words, boxw, boxh, font_size, lespc, lispc):
    '''将text按照行分割后，返回每一行的数据

    Returns:
        list: [Line ...] 如果列表为空，表示不能按照指定配置在文本框内完成排版
    '''
    texth = 0
    lines = []
    prei, i = 0, 0
    line = Line(font, lespc)
    while i < len(words):
        word = words[i]
        line.append(word)
        lw = line.get_display_width()
        # print('\tline width', lw, line, i, prei)
        if lw > boxw:
            # 超框了直接返回
            if i == prei:
                return []
            # 更新文本高V度
            texth += font_size
            if lines:
                texth += lispc
            # 判断行高是否超限
            if texth > boxh:
                return []
            # 添加新行
            line.pop()
            lines.append(line)
            line = Line(font, lespc)
            prei = i
            # 如果行首有违反排版规则的字符则从前面的行借字符
            if word[0] in _PUNCTUATION_BLOCK_SET:
                prei -= 1
                i = prei
                lines[-1].pop()
        else:
            i += 1
    if line.words:
        # 更新文本高V度
        texth += font_size
        if lines:
            texth += lispc
        # 判断行高是否超限
        if texth > boxh:
            return []
        lines.append(line)

    return lines


def _build_max_font_lines(text, section):
    '''在文本框内寻找能最大利用文本框显示区域字号，并执行分行操作'''
    # 1. 把文本块中所有的单词和数字找出来，保证他们不会被分割。这样符合排版规则
    words = _splite_text_to_words(text)
    # 3. 求字号范围
    boxw, boxh = section.box.w, section.box.h
    if section.dir == 'v':
        boxw, boxh = boxh, boxw
    max_font_size = int(min(boxw, boxh))
    min_font_size = int(min(boxw, boxh, _MIN_FONT_SIZE))
    # 4. 二分法查找最合适的字号分行操作
    lfs, rfs = min_font_size, max_font_size
    lines = []
    while lfs <= rfs:
        mfs = lfs + int((rfs - lfs) / 2)
        lespc = int(section.letter_spacing_factor * mfs)
        lispc = int(section.line_spacing_factor * mfs)
        # print('fontsize', mfs, lfs, rfs)
        font = _get_font(section.font, mfs)
        lines = _build_lines(text, font, words, boxw, boxh, mfs, lespc, lispc)
        if mfs == lfs:
            break
        if lines:
            lfs = mfs
        else:
            rfs = mfs
    return lines, font, mfs, lispc, lespc


def _build_trimed_lines(text, section):
    if section.dir == 'h':
        fs = section.box.h
        width = section.box.w
    else:
        fs = section.box.w
        width = section.box.h

    lespc = int(section.letter_spacing_factor * fs)
    lispc = int(section.line_spacing_factor * fs)
    font = _get_font(section.font, fs)
    line = Line(font, lespc)

    limit = width - font.getsize('…')[0]
    i = 0
    while i < len(text):
        line.append(text[i])
        if line.get_display_width() > limit:
            break
        i += 1
    if i < len(text):
        line.pop()
        line.append('…')

    return [line], font, fs, lispc, lespc


def layout_text(text, section) -> Layout:
    '''按照 section 指定的配置对 text 进行排版

    Args:
        text (str): 待排版字体
        section (config.Section): 排版配置

    Returns:
        Layout: 排版好的 Layout 对象
    '''
    # 按规则执行分行
    if section.trim:
        lines, font, font_size, line_spacing, letter_spacing = _build_trimed_lines(
            text, section)
    else:
        lines, font, font_size, line_spacing, letter_spacing = _build_max_font_lines(
            text, section)
    # 进行布局运算
    ret = Layout(lines, font, font_size, line_spacing, letter_spacing)
    ret.update(section.box, section.dir, section.valign, section.halign)
    return ret
