import glob
import os
import shutil
import sys
import threading
import time
import configparser
import eventlet
from tkinter import messagebox, Tk, filedialog, colorchooser

import cv2
import numpy as np
from PIL import Image
from PyQt6 import QtWidgets, QtCore, QtGui

from translate import translate, change_translate_mod
from covermaker import conf, render
from inpainting import Inpainting
from interface import Ui_MainWindow
from characterStyle import Ui_Dialog as CharacterStyleDialog
from textblockdetector import dispatch as textblockdetector
from utils import compute_iou

# tkinter弹窗初始化
root = Tk()
root.withdraw()
# 超时跳出
eventlet.monkey_patch()


# 重定向控制台信号
class Shell(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


# 程序参数
class var:
    img_language = 'ja'  # 原始语言
    word_language = 'zh-CN'  # 翻译语言
    word_way = 0  # 文字输出方向
    word_conf = conf.Section()  # 文字渲染参数
    word_mod = 'auto'  # 文字定位模式
    img_re_bool = True  # 图像修复开关
    img_re_mod = 1  # 图像修复模式


# 运行中的缓存文件
class memory():
    model = None  # 模型
    img_show = None  # 显示的图像
    img_mark = None  # 文字掩码
    img_mark_more = None  # 文字掩码2
    img_repair = None  # 修复后的图像
    img_textlines = []  # 掩码box
    textline_box = []  # 范围内的box

    img_in = None  # 输入图像
    img_out = None  # 输出图像

    task_out = ''  # 导出的目录
    task_name = []  # 文件名
    task_img = []  # 图片原文件

    action_save_num = 0  # 行为记录
    action_save_img = []  # 存档
    range_choice = [0, 0, 0, 0]  # 当前选中的范围


# 运行状态
class state():
    mod_ready = False  # 模型状态
    action_running = False  # 运行状态
    text_running = False  # 是否是文字输出
    img_half = False  # 当前图片缩小一半
    task_num = 0  # 任务数量
    task_end = 0  # 完成数量
    ttsing = False  # 语音输出锁(未使用多线程)


# 主程序
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.setWindowIcon(QtGui.QIcon('ico.png'))
        self.ui.setupUi(self)

        self.var = var()
        self.state = state()
        self.memory = memory()

        sys.stdout = Shell(newText=self.shelltext)  # 下面将输出重定向到textEdit中
        print('这里是控制台')
        self.uireadly()  # 初始化按钮槽
        self.thredstart()  # 开始线程

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):  # 重写移动事件
        if self._tracking:
            self._endPos = e.globalPosition().toPoint() - self._startPos
            self.move(self._winPos + self._endPos)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._winPos = self.pos()
            self._startPos = QtCore.QPoint(e.globalPosition().toPoint())
            self._tracking = True

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None

    # 读取图像，解决imread不能读取中文路径的问题
    def cv2_imread(self, path):
        img = Image.open(path)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return img

    # 控制台输出到text
    def shelltext(self, text):
        if text!='\n':
            self.ui.textEdit_3.append(text)
            self.ui.textEdit_3.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    # 槽
    def uireadly(self):
        self.ui.action1.triggered.connect(
            lambda event: QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://sljly.xyz/')))
        self.ui.action2.triggered.connect(
            lambda event: QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://github.com/jtl1207/comic-translation/')))
        # self.ui.action3.triggered.connect(lambda event: print('1111'))

        self.ui.actionja.triggered.connect(lambda event: self.change_mod('ja'))
        self.ui.actionen.triggered.connect(lambda event: self.change_mod('en'))
        self.ui.actionko.triggered.connect(lambda event: self.change_mod('ko'))
        self.ui.actioncn.triggered.connect(lambda event: self.change_out_language('cn'))
        self.ui.actionen_2.triggered.connect(lambda event: self.change_out_language('en'))
        self.ui.actionKorean.triggered.connect(lambda event: self.change_out_language('ko'))

        self.ui.actionin_imgs.triggered.connect(lambda event: self.change_img(True))
        self.ui.actionin_img.triggered.connect(lambda event: self.change_img(False))
        # self.actionfont = QtWidgets.QWidgetAction(self.menuBar)
        # self.actionfont.setObjectName('actionfont')
        # self.actionfont.setText("导入字体")
        # self.menuBar.addAction(self.actionfont)
        self.ui.actionfont.triggered.connect(lambda event: self.change_font())

        self.ui.pushButton_2.clicked.connect(lambda event: self.change_word_way())
        self.ui.pushButton_13.clicked.connect(lambda event: self.change_word_mod())
        self.ui.pushButton_16.clicked.connect(lambda event: self.new_character_style_window())
        self.ui.pushButton_8.clicked.connect(lambda event: self.change_img_re())
        self.ui.pushButton_11.clicked.connect(lambda event: self.change_img_mod())

        self.ui.pushButton_4.clicked.connect(lambda event: self.translation_img())
        self.ui.pushButton_14.clicked.connect(lambda event: self.text_add())
        self.ui.pushButton_12.clicked.connect(lambda event: self.text_clean())
        self.ui.pushButton_9.clicked.connect(lambda event: self.auto_text_clean())
        self.ui.pushButton_10.clicked.connect(lambda event: self.auto_translation())
        self.ui.pushButton_7.clicked.connect(lambda event: self.cancel())
        self.ui.pushButton_6.clicked.connect(lambda event: self.save())

        self.ui.pushButton.clicked.connect(lambda event: self.tts())
        self.ui.pushButton_3.clicked.connect(lambda event: self.change_translate_mod())
        self.ui.pushButton_5.clicked.connect(lambda event: self.doit())
        self.ui.pushButton_15.clicked.connect(lambda event: self.closeit())

    # 其他线程
    def thredstart(self):
        QtCore.QTimer.singleShot(500, self.config_read)
        QtCore.QTimer.singleShot(1000, self.thred_cuda)
        QtCore.QTimer.singleShot(1500, self.thread_net)

    # 检测cuda状态
    def thred_cuda(self):
        try:
            import paddle
            if paddle.device.get_device() == 'cpu':
                print('paddle:cuda异常,cpu模式')
                self.ui.label_10.setText('cpu')
            elif paddle.device.get_device() == 'gpu:0':
                print(f'paddle:cuda正常')
        except:
            print('Error:paddle异常')
            self.ui.label_10.setText('异常')
        try:
            import torch
            if torch.cuda.is_available():
                print("pytorch:cuda正常")
            else:
                print("pytorch:cuda异常,cpu模式")
                self.ui.label_10.setText('cpu')
        except:
            print('Error:pytorch异常')
            self.ui.label_10.setText('异常')
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                print("tensorflow:cuda正常")
            else:
                print("tensorflow:cuda异常,cpu模式")
                self.ui.label_10.setText('cpu')
        except:
            print('Error:tensorflow异常')
            self.ui.label_10.setText('异常')

        if self.ui.label_10.text() == '检测中':
            self.ui.label_10.setText('正常')

    # 检测网络状态
    def thread_net(self):
        t = time.time()
        try:
            with eventlet.Timeout(20, False):
                text = translate("hello", "zh-CN", "auto", in_mod=3)
            if text != '你好':
                print('google翻译:网络异常,不推荐使用代理')
            else:
                print(f'google翻译:网络正常,ping:{(time.time() - t) * 1000:.0f}ms')
        except:
            print('google翻译:网络异常,不推荐使用代理')

        t = time.time()
        try:
            with eventlet.Timeout(20, False):
                text = translate("hello", "zh-CN", "auto", in_mod=1)
            if text != '你好':
                print('deepl翻译:网络异常,不推荐使用代理')
            else:
                print(f'deepl翻译:网络正常,ping:{(time.time() - t) * 1000:.0f}ms')
        except:
            print('deepl翻译:网络异常,不推荐使用代理')

        from gtts.tts import gTTS
        import pyglet
        try:
            tts = gTTS(text='お兄ちゃん大好き', lang='ja')
            filename = 'temp.mp3'
            tts.save(filename)
            music = pyglet.media.load(filename, streaming=False)
            music.play()
            time.sleep(music.duration)
            os.remove(filename)
            print(f'TTS:网络正常')
        except:
            print('TTS:网络异常,不推荐使用代理')


    # 切换语言
    def change_mod(self, language):
        self.ui.actionja.setChecked(False)
        self.ui.actionen.setChecked(False)
        self.ui.actionko.setChecked(False)
        if language == 'ja':
            thread_language = threading.Thread(target=self.thread_language('ja'))
        elif language == 'en':
            thread_language = threading.Thread(target=self.thread_language('en'))
        elif language == 'ko':
            thread_language = threading.Thread(target=self.thread_language('ko'))
        thread_language.setDaemon(True)
        thread_language.start()
        print(f'Info:切换检测语言{language}')
        self.config_save('img_language', language)

    def thread_language(self, language):
        self.state.mod_ready = False
        self.ui.label_4.setText('未加载')
        if language == 'ja':
            from manga_ocr.ocr import MangaOcr
            self.memory.model = MangaOcr()
            self.ui.actionja.setChecked(True)
        elif language == 'en':
            import paddleocr
            self.memory.model = paddleocr.PaddleOCR(
                show_log=False,  # 禁用日志
                use_gpu=True,  # 使用gpu
                cls=False,  # 角度分类
                det_limit_side_len=320,  # 检测算法前向时图片长边的最大尺寸，
                det_limit_type='max',  # 限制输入图片的大小,可选参数为limit_type[max, min] 一般设置为 32 的倍数，如 960。
                ir_optim=False,
                use_fp16=False,  # 16位半精度
                use_tensorrt=False,  # 使用张量
                gpu_mem=6000,  # 初始化占用的GPU内存大小
                cpu_threads=20,
                enable_mkldnn=True,  # 是否启用mkldnn
                max_batch_size=512,  # 图片尺寸最大大小
                cls_model_dir='paddleocr/model/cls',
                # cls模型位置
                # image_dir="",  # 通过命令行调用时间执行预测的图片或文件夹路径
                det_algorithm='DB',  # 使用的检测算法类型DB/EAST
                det_model_dir='paddleocr/model/det/det_infer',
                # 检测模型所在文件夹。传参方式有两种，1. None: 自动下载内置模型到 ~/.paddleocr/det；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件
                # DB(还有east,SAST)
                det_db_thresh=0.3,  # DB模型输出预测图的二值化阈值
                det_db_box_thresh=0.6,  # DB模型输出框的阈值，低于此值的预测框会被丢弃
                det_db_unclip_ratio=1.3,  # DB模型输出框扩大的比例
                use_dilation=True,  # 缩放图片
                det_db_score_mode="fast",  # 计算分数模式,fast对应原始的rectangle方式，slow对应polygon方式。
                # 文本识别器的参数
                rec_algorithm='CRNN',  # 使用的识别算法类型
                rec_model_dir='paddleocr/model/rec/ch_rec_infer',
                # 识别模型所在文件夹。传承那方式有两种，1. None: 自动下载内置模型到 ~/.paddleocr/rec；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件
                # rec_image_shape="3,32,320",  # 识别算法的输入图片尺寸
                # cls_batch_num=36,  #
                # cls_thresh=0.9,  #
                lang='ch',  # 语言(这个用的是中英模型)
                det=True,  # 检测文字位置
                rec=True,  # 识别文字内容
                use_angle_cls=False,  # 识别竖排文字
                rec_batch_num=36,  # 进行识别时，同时前向的图片数
                max_text_length=30,  # 识别算法能识别的最大文字长度
                # rec_char_dict_path='',  # 识别模型字典路径，当rec_model_dir使用方自己模型时需要
                use_space_char=True,  # 是否识别空格
            )
            self.ui.actionen.setChecked(True)
        elif language == 'ko':
            import paddleocr
            self.memory.model = paddleocr.PaddleOCR(
                # show_log=False, #禁用日志
                use_gpu=True,  # 使用gpu
                cls=False,  # 角度分类
                det_limit_side_len=320,  # 检测算法前向时图片长边的最大尺寸，
                det_limit_type='max',  # 限制输入图片的大小,可选参数为limit_type[max, min] 一般设置为 32 的倍数，如 960。
                ir_optim=False,
                use_fp16=False,  # 16位半精度
                use_tensorrt=False,  # 使用张量
                gpu_mem=6000,  # 初始化占用的GPU内存大小
                cpu_threads=20,
                enable_mkldnn=True,  # 是否启用mkldnn
                max_batch_size=512,  # 图片尺寸最大大小
                cls_model_dir='paddleocr/model/cls',
                # cls模型位置
                # image_dir="",  # 通过命令行调用时间执行预测的图片或文件夹路径
                det_algorithm='DB',  # 使用的检测算法类型DB/EAST
                det_model_dir='paddleocr/model/det/det_infer',
                # 检测模型所在文件夹。传参方式有两种，1. None: 自动下载内置模型到 ~/.paddleocr/det；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件
                # DB(还有east,SAST)
                det_db_thresh=0.3,  # DB模型输出预测图的二值化阈值
                det_db_box_thresh=0.6,  # DB模型输出框的阈值，低于此值的预测框会被丢弃
                det_db_unclip_ratio=1.3,  # DB模型输出框扩大的比例
                use_dilation=True,  # 缩放图片
                det_db_score_mode="fast",  # 计算分数模式,fast对应原始的rectangle方式，slow对应polygon方式。
                # 文本识别器的参数
                rec_algorithm='CRNN',  # 使用的识别算法类型
                rec_model_dir='paddleocr/model/rec/ko_rec_infer',
                # 识别模型所在文件夹。传承那方式有两种，1. None: 自动下载内置模型到 ~/.paddleocr/rec；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件
                # rec_image_shape="3,32,320",  # 识别算法的输入图片尺寸
                # cls_batch_num=36,  #
                # cls_thresh=0.9,  #
                lang='korean',  # 语言
                det=True,  # 检测文字位置
                rec=True,  # 识别文字内容
                use_angle_cls=False,  # 识别竖排文字
                rec_batch_num=36,  # 进行识别时，同时前向的图片数
                max_text_length=30,  # 识别算法能识别的最大文字长度
                # rec_char_dict_path='',  # 识别模型字典路径，当rec_model_dir使用方自己模型时需要
                use_space_char=True,  # 是否识别空格
            )
            self.ui.actionko.setChecked(True)
        self.state.mod_ready = True
        self.ui.label_4.setText(f'{language}')
        self.var.img_language = language

    def change_out_language(self, language):
        self.ui.actioncn.setChecked(False)
        self.ui.actionen_2.setChecked(False)
        self.ui.actionKorean.setChecked(False)
        if language == 'cn':
            self.var.word_language = 'zh-CN'
            self.ui.actioncn.setChecked(True)
        elif language == 'en':
            self.var.word_language = 'en'
            self.ui.actionen_2.setChecked(True)
        elif language == 'ko':
            self.var.word_language = 'ko'
            self.ui.actionKorean.setChecked(True)
        print(f'Info: 输出语言{self.var.word_language}')
        self.config_save('word_language', self.var.word_language)

    # 读取图片
    def change_img(self, s):
        if self.state.task_num != self.state.task_end:
            if not messagebox.askyesno('提示', '当前有任务执行中,是否清空队列?'):
                return
        self.state.task_num = 0
        self.state.task_end = 0
        self.memory.task_out = ''
        self.memory.task_name = []
        self.memory.task_img = []

        if s:
            path = filedialog.askdirectory()
            if path == '':
                return
            files = []
            for ext in (
                    '*.BMP', '*.DIB', '*.JPEG', '*.JPG', '*.JPE', '*.PNG', '*.PBM', '*.PGM', '*.PPMSR', '*.RAS',
                    '*.TIFF',
                    '*.TIF', '*.EXR', '*.JP2', '*.WEBP'):
                files.extend(glob.glob(os.path.join(path, ext)))
            files.sort(key=lambda x: int("".join(list(filter(str.isdigit, x)))))  # 文件名按数字排序
            self.memory.task_out = os.path.dirname(path) + '/out/'
            for file_path in files:
                try:
                    try:
                        img = cv2.imread(file_path)
                        height, width, channel = img.shape
                    except:
                        img = self.cv2_imread(file_path)
                        height, width, channel = img.shape
                    if channel == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    self.memory.task_img.append(img)
                    self.state.task_num += 1
                    self.memory.task_name.append(os.path.basename(file_path))
                except:
                    messagebox.showerror(title='Error', message=f'{file_path}图片读取Error')
            if self.state.task_num == 0:
                self.panel_clean()
                print(f'War:未检测到图片')
            else:
                self.panel_shownext()
                print(f'Info:成功导入{self.state.task_num}张图片')
        else:
            filetypes = [("支持格式",
                          "*.BMP;*.DIB;*.JPEG;*.JPG;*.JPE;*.PNG;*.PBM;*.PGM;*.PPMSR;*.RAS','.TIFF','.TIF;*.EXR;*.JP2;*.WEBP")]
            path = filedialog.askopenfilename(title='选择单张图片', filetypes=filetypes)
            if path == '':
                return
            root, ext = os.path.splitext(os.path.basename(path))
            try:
                try:
                    img = cv2.imread(path)
                    height, width, channel = img.shape
                except:
                    img = self.cv2_imread(path)
                    height, width, channel = img.shape
                if channel == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                self.memory.task_img.append(img)
                self.state.task_num = 1
                self.state.task_end = 0
                self.memory.task_out = os.path.dirname(path)
                self.memory.task_name = []
                self.memory.task_name.append(f'{root}_re{ext}')
                self.panel_shownext()
                print(f'Info:成功导入{self.state.task_num}张图片')
            except:
                messagebox.showerror(title='Error', message=f'{path}图片读取Error')
                self.state.task_num = 0
                self.panel_clean()
        if self.state.task_num > 0:
            self.ui.img.flag_switch = True
            self.ui.pushButton_4.setEnabled(True)
            self.ui.pushButton_14.setEnabled(True)
            self.ui.pushButton_12.setEnabled(True)
            self.ui.pushButton_9.setEnabled(True)
            self.ui.pushButton_6.setEnabled(True)

    # 读取字体
    def change_font(self):
        filetypes = [("支持格式", "*.TTF;*.TTC;*.OTF")]
        path = filedialog.askopenfilename(title='选择字体', filetypes=filetypes, initialdir='./covermaker/fonts')
        if path == '':
            return
        else:
            if not os.path.exists(f'covermaker/fonts/{os.path.basename(path)}'):
                shutil.copyfile(f'{path}', f'covermaker/fonts/{os.path.basename(path)}')
            self.var.word_conf.font = f'{os.path.basename(path)}'
            self.ui.label_6.setText(f'{os.path.basename(path)}')
        self.config_save('font', self.var.word_conf.font)

    # 清空面板
    def panel_clean(self):
        self.ui.img.clear()
        self.ui.img.setFixedWidth(450)
        self.ui.img.setFixedHeight(696)
        self.ui.img.setText(
            '1.左上角导入漫画,选择要翻译的语言,输出的字体\n2.自动翻译或手动选择区域\n3.手动翻译需要在右下角确定\n建议提前排好图片顺序与名称\n\n\n环境要求:\n需要网络,使用Google翻译(中国)\n有显卡可显著加快图像处理速度(cuda)\n需要驱动版本>=472.50\n正常处理图片时间1s/张\n\n\n支持的图片格式:\nWindows位图文件 - BMP, DIB\nJPEG文件 - JPEG, JPG, JPE\n便携式网络图片 - PNG\n便携式图像格式 - PBM，PGM，PPM\nSun rasters - SR，RAS\nTIFF文件 - TIFF，TIF\nOpenEXR HDR 图片 - EXR\nJPEG 2000 图片- jp2')
        self.ui.img.setStyleSheet('background-color:rgb(255,255,255);\ncolor:rgba(0,0,0,255);')
        self.ui.label_3.setText(f'{self.state.task_end}/{self.state.task_num}')
        self.memory.action_save_num = 0
        self.memory.action_save_img = []

    # 更新面板
    def panel_shownext(self):
        self.ui.img.setStyleSheet('background-color:rgb(255,255,255);\ncolor:rgba(0,0,0,0);')
        img = self.memory.task_img[self.state.task_end]
        self.memory.img_show = img.copy()
        self.memory.img_mark_more, self.memory.img_mark, self.memory.img_textlines = textblockdetector(img)
        self.memory.img_mark_more[self.memory.img_mark_more != 0] = 255

        height, width, channel = img.shape
        self.state.img_half = False
        if height > 900 or width > 1500:
            self.state.img_half = True
            height //= 2
            width //= 2
        else:
            self.state.img_half = False
        self.ui.img.setFixedWidth(width)
        self.ui.img.setFixedHeight(height)

        self.show_img()
        self.ui.label_3.setText(f'{self.state.task_end}/{self.state.task_num}')
        self.memory.img_repair = None
        self.memory.action_save_num = 0
        self.memory.action_save_img = []

    # 保存图片
    def save(self):
        self.state.action_running = False
        if not os.path.exists(self.memory.task_out):
            os.mkdir(self.memory.task_out)
        name = self.memory.task_out + "/" + self.memory.task_name[self.state.task_end]
        cv2.imwrite(name, self.memory.img_show)
        self.state.task_end += 1
        self.ui.img.update()

        messagebox.showinfo(title='成功', message=f'图片保存完成\n{self.memory.task_out}\\{name}')
        self.ui.textEdit_3.setText('')
        print(f'Info:图片保存完成\n{name}')

        if self.state.task_end < self.state.task_num:
            self.panel_shownext()
        else:
            self.panel_clean()
            self.ui.img.flag_switch = False  # 矩形绘制锁
            self.ui.pushButton_4.setEnabled(False)
            self.ui.pushButton_14.setEnabled(False)
            self.ui.pushButton_12.setEnabled(False)
            self.ui.pushButton_9.setEnabled(False)
            self.ui.pushButton_7.setEnabled(False)
            self.ui.pushButton_6.setEnabled(False)
            self.ui.pushButton_5.setEnabled(False)
            self.ui.pushButton_15.setEnabled(False)
            self.ui.pushButton.setEnabled(False)
            self.ui.pushButton_3.setEnabled(False)

    # 输出文字方向
    def change_word_way(self):
        if self.var.word_way == 1:
            self.var.word_way = 2
            self.ui.pushButton_2.setText('排列:横向')
            print('Info:文字横向输出')
        else:
            self.var.word_way = 1
            self.ui.pushButton_2.setText('排列:垂直')
            print('Info:文字垂直输出')
        self.config_save('word_way', self.var.word_way)

    # 文字定位模式
    def change_word_mod(self):
        if self.var.word_mod == 'auto':
            self.var.word_mod = 'Handmade'
            print('Info:文字定位模式:手动')
            self.ui.pushButton_13.setText('定位:手动')
        else:
            self.var.word_mod = 'auto'
            print('Info:文字定位模式:自动')
            self.ui.pushButton_13.setText('定位:自动')
        self.config_save('word_mod', self.var.word_mod)

    # 字距设置
    def new_character_style_window(self):
        Window = CharacterStyle()
        Window.ui.pushButton_1.setStyleSheet(
            f'background-color: {self.var.word_conf.color};border-width:0px;border-radius:11px;')
        Window.ui.lineEdit_3.setText(str(self.var.word_conf.stroke_width))
        Window.ui.pushButton_3.setStyleSheet(
            f'background-color: {self.var.word_conf.stroke_fill};border-width:0px;border-radius:11px;')
        Window.ui.lineEdit.setText(str(self.var.word_conf.letter_spacing_factor))
        Window.ui.lineEdit_2.setText(str(self.var.word_conf.line_spacing_factor))
        Window.stroke_fill = self.var.word_conf.stroke_fill
        Window.color = self.var.word_conf.color
        Window.exec()
        if Window.re[0]:
            self.var.word_conf.letter_spacing_factor = Window.re[1]
            self.var.word_conf.line_spacing_factor = Window.re[2]
            self.var.word_conf.color = Window.re[3]
            self.var.word_conf.stroke_width = Window.re[4]
            self.var.word_conf.stroke_fill = Window.re[5]
            print(f'Info:字距{Window.re[1]}\n文字颜色{Window.re[3]}\n行距{Window.re[2]}\n阴影颜色{Window.re[5]}\n阴影宽度{Window.re[4]}')
            self.config_save('line_spacing_factor', self.var.word_conf.line_spacing_factor)
            self.config_save('letter_spacing_factor', self.var.word_conf.letter_spacing_factor)
            self.config_save('stroke_fill', self.var.word_conf.stroke_fill)
            self.config_save('color', self.var.word_conf.color)
            self.config_save('stroke_width', self.var.word_conf.stroke_width)
        Window.destroy()

    # 图像修复开关
    def change_img_re(self):
        if self.var.img_re_bool:
            self.var.img_re_bool = False
            self.ui.pushButton_8.setText('停用')
            print('Info:图像修复关闭')
            print(' 图像修复模式:背景抹白')
        else:
            self.var.img_re_bool = True
            self.ui.pushButton_8.setText('启用')
            print('Info:图像修复打开')
            if self.var.img_re_mod == 1:
                print(' 图像修复模式:标准文字修复')
            elif self.var.img_re_mod == 2:
                print(' 图像修复模式:标准文字修复膨胀1')
            elif self.var.img_re_mod == 3:
                print(' 图像修复模式:标准文字修复膨胀2')
            elif self.var.img_re_mod == 4:
                print(' 图像修复模式:增强文字修复')
            elif self.var.img_re_mod == 5:
                print(' 图像修复模式:增强文字修复膨胀1')
            elif self.var.img_re_mod == 6:
                print(' 图像修复模式:增强文字修复膨胀2')
        self.config_save('img_re_bool', self.var.img_re_bool)

    # 图像修复模式
    def change_img_mod(self):
        if self.var.img_re_mod == 6:
            self.var.img_re_mod = 1
        else:
            self.var.img_re_mod += 1
        if self.var.img_re_mod == 1:
            print('Info:图像修复模式:标准文字修复')
        elif self.var.img_re_mod == 2:
            print('Info:图像修复模式:标准文字修复膨胀1')
        elif self.var.img_re_mod == 3:
            print('Info:图像修复模式:标准文字修复膨胀2')
        elif self.var.img_re_mod == 4:
            print('Info:图像修复模式:增强文字修复')
        elif self.var.img_re_mod == 5:
            print('Info:图像修复模式:增强文字修复膨胀1')
        elif self.var.img_re_mod == 6:
            print('Info:图像修复模式:增强文字修复膨胀2')
        self.memory.img_repair = None
        self.config_save('img_re_mod', self.var.img_re_mod)

    def doit(self):
        if self.state.action_running:
            self.action_save()
            if self.state.text_running:
                self.do_add_text()
            else:
                self.do_translation()


    def do_translation(self):
        pos = self.memory.textline_box[0]
        if self.var.img_re_bool:
            if self.memory.img_repair is None:
                self.img_repair()
            roi = self.memory.img_repair[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
            self.memory.img_show[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]] = roi
        else:
            white = np.zeros([pos[3], pos[2], 3], dtype=np.uint8) + 255
            self.memory.img_show[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]] = white

        print('Info:图像修复完成')
        # 添加文字
        text = self.ui.textEdit_2.toPlainText()
        if text.replace(" ", "") != '':
            img = self.memory.img_show.copy()

            pos = self.memory.textline_box[0]
            if pos is None: print('Error:boxError')
            self.var.word_conf.box = conf.Box(pos[0], pos[1], pos[2], pos[3])
            if self.var.word_way == 2 or self.var.word_language == 'en' or self.var.word_language == 'ko':
                if self.var.word_way == 1:
                    print('War:当前语言不支持竖排文字')
                self.var.word_conf.dir = 'h'
            else:
                self.var.word_conf.dir = 'v'

            img = render.Render(img)
            img = img.draw(text, self.var.word_conf)

            self.memory.img_show = img.copy()
        else:
            print('War:未输入文字')
        self.show_img()
        del (self.memory.textline_box[0])

        if len(self.memory.textline_box) == 0:
            self.state.action_running = False
            self.ui.pushButton_5.setEnabled(False)
            self.ui.pushButton.setEnabled(False)
            self.ui.pushButton_3.setEnabled(False)
            self.ui.pushButton_15.setEnabled(False)
            self.ui.textEdit.setText('')
            self.ui.textEdit_2.setText('')
        else:
            box = self.memory.textline_box[0]
            result = self.memory.model(self.memory.img_show[box[1]:box[3] + box[1], box[0]:box[2] + box[0]])
            self.ui.textEdit.setText(result)
            if result.replace(" ", "") == '':
                print('War:文字识别异常,请手动输入')
                self.ui.textEdit_2.setText('')
            else:
                with eventlet.Timeout(20, False):
                    self.ui.textEdit_2.setText(translate(result, f'{self.var.word_language}', "auto"))
                if self.ui.textEdit_2.toPlainText() == '':
                    self.ui.textEdit_2.setText('翻译超时')

    def do_add_text(self):
        text = self.ui.textEdit_2.toPlainText()
        if text.replace(" ", "") != '':
            img = self.memory.img_show.copy()

            pos = self.memory.textline_box[0]
            if pos is None: print('Error:boxError')
            # elif self.state.img_half:
            #     pos = [i * 2 for i in pos]
            self.var.word_conf.box = conf.Box(pos[0], pos[1], pos[2], pos[3])

            if self.var.word_way == 2 or self.var.word_language == 'en' or self.var.word_language == 'ko':
                if self.var.word_way == 1:
                    print('War:当前语言不支持竖排文字')
                self.var.word_conf.dir = 'h'
            else:
                self.var.word_conf.dir = 'v'
            img = render.Render(img)
            img = img.draw(text, self.var.word_conf)
            self.memory.img_show = img
            # 显示图像
            self.show_img()
        else:
            print('War:未输入文字')
        self.ui.textEdit.setText('')
        self.ui.textEdit_2.setText('')
        self.state.text_running = self.state.action_running = False
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_15.setEnabled(False)
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_3.setEnabled(False)

    def closeit(self):
        self.state.action_running = False
        self.ui.textEdit.setText('')
        self.ui.textEdit_2.setText('')
        self.state.action_running = False
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_15.setEnabled(False)
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_3.setEnabled(False)

    # 翻译选中内容
    def translation_img(self):
        if not self.state.mod_ready:
            print('Error:模型未正确加载')
            return
        if not self.state.action_running:
            pos = self.get_pos()
            if pos is None:
                print('Error:boxError')
                return
            textline_box = []
            self.memory.textline_box = []

            for i in self.memory.img_textlines:
                if compute_iou([i.xyxy[0], i.xyxy[1], i.xyxy[2], i.xyxy[3]],
                               [pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]) > 0.6:
                    textline_box.append([i.xyxy[0], i.xyxy[1], i.xyxy[2] - i.xyxy[0] + 3, i.xyxy[3] - i.xyxy[1]])

            if len(textline_box) == 0:
                self.memory.textline_box.append(pos)
                box = pos
                print('War:文字位置检测异常\n推荐使用加强版图像修复(或抹白)')
            elif len(textline_box) == 1:
                box = pos
                if self.var.word_mod == 'Handmade':
                    self.memory.textline_box.append(pos)
                else:
                    self.memory.textline_box.append(textline_box[0])
                print('Info:检测成功,请确认翻译')
            elif len(textline_box) > 1:
                for i in textline_box:
                    self.memory.textline_box.append(i)
                box = textline_box[0]
                print('Info:当前区域检测有多段文字\n文字输出排列强制自动\n请多次确认翻译')

            result = self.memory.model(self.memory.img_show[box[1]:box[3] + box[1], box[0]:box[2] + box[0]])
            if self.var.img_language == 'ja':
                self.ui.textEdit.setText(result)
            else:
                str = ''
                for i in result[1]:
                    str = str + i[0]
                result = str
                self.ui.textEdit.setText(result)
            if result.replace(" ", "") == '':
                print('Info:文字识别异常,请手动输入')
                self.ui.textEdit_2.setText('')
            else:
                with eventlet.Timeout(20, False):
                    self.ui.textEdit_2.setText(translate(result, f'{self.var.word_language}', "auto"))
                if self.ui.textEdit_2.toPlainText() =='':
                    self.ui.textEdit_2.setText('翻译超时')

            self.state.action_running = True
            self.ui.pushButton_5.setEnabled(True)
            self.ui.pushButton_15.setEnabled(True)
            self.ui.pushButton.setEnabled(True)
            self.ui.pushButton_3.setEnabled(True)
        else:
            print('War:任务队列未完成,右下角继续')

    def text_add(self):
        if not self.state.action_running:
            pos = self.get_pos()
            if pos is None: return
            self.action_save()
            self.memory.textline_box = []
            self.memory.textline_box.append(pos)

            self.ui.textEdit.setText('下方输入文字')
            # self.ui.textEdit_2.setText('')
            self.state.action_running = True
            self.ui.pushButton_5.setEnabled(True)
            self.ui.pushButton_15.setEnabled(True)
            self.state.text_running = True
        else:
            print('War:任务队列未完成,右下角继续')

    def text_clean(self):
        if not self.state.action_running:
            pos = self.get_pos()
            if pos is None: return
            self.action_save()
            text = 0
            for i in self.memory.img_textlines:
                if compute_iou([i.xyxy[0], i.xyxy[1], i.xyxy[2], i.xyxy[3]],
                               [pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]) > 0.6:
                    text += 1
            if text == 0:
                print('War:当前区域文字检测异常\n推荐使用加强版图像修复(或抹白)')
            # 图像修复
            if self.var.img_re_bool:
                if self.memory.img_repair is None:
                    self.img_repair()
                roi = self.memory.img_repair[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
                self.memory.img_show[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]] = roi
            else:
                white = np.zeros([pos[3], pos[2], 3], dtype=np.uint8) + 255
                self.memory.img_show[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]] = white
            print('Info:图像修复完成')
            # 显示图像
            self.show_img()
        else:
            print('War:任务队列未完成,右下角继续')

    def auto_text_clean(self):
        if not self.state.action_running:
            self.action_save()
            # 图像修复
            if self.memory.img_repair is None:
                self.img_repair()
            self.memory.img_show = self.memory.img_repair.copy()
            print('Info:图像修复完成\n部分区域需要自行抹白')
            # 显示图像
            self.show_img()
        else:
            print('War:任务队列未完成,右下角继续')

    # 提取box
    def get_pos(self):
        pos = self.memory.range_choice = self.ui.img.img_pos
        if pos == [0, 0, 0, 0] or pos[2] < 2 or pos[3] < 2:
            print('Error:未选择输入区域')
            return None
        if self.state.img_half:
            pos = self.memory.range_choice = [pos[0] * 2, pos[1] * 2, pos[2] * 2, pos[3] * 2]
        return pos

    # 显示图像
    def show_img(self):
        if self.state.img_half:
            height, width, channel = self.memory.img_show.shape
            height //= 2
            width //= 2
            img = cv2.resize(self.memory.img_show, (width, height))
        else:
            img = self.memory.img_show
        cv2.imwrite('save.jpg',self.memory.img_show)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * img.shape[2],
                                 QtGui.QImage.Format.Format_RGB888)
        self.ui.img.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 撤销
    def cancel(self):
        if not self.state.action_running:
            self.memory.img_show = self.memory.action_save_img[self.memory.action_save_num - 1].copy()
            self.memory.action_save_num -= 1
            self.show_img()
            print('Info:撤销完成')
            if self.memory.action_save_num == 0:
                self.ui.pushButton_7.setEnabled(False)
        else:
            print('War:任务队列未完成,右下角继续')

    # 保存
    def action_save(self):
        if len(self.memory.action_save_img) == self.memory.action_save_num:
            self.memory.action_save_img.append(self.memory.img_show.copy())
        else:
            self.memory.action_save_img[self.memory.action_save_num] = self.memory.img_show.copy()
        self.memory.action_save_num += 1
        if self.memory.action_save_num > 0:
            self.ui.pushButton_7.setEnabled(True)

    # 图像修复,送入网络模型
    def img_repair(self):
        print('Info:检测中,请稍后')
        if self.var.img_re_mod < 4:
            mark = self.memory.img_mark
        else:
            mark = self.memory.img_mark_more
        if self.var.img_re_mod % 3 != 1:
            kernel = np.ones((5, 5), dtype=np.uint8)
            mark = cv2.dilate(mark, kernel, self.var.img_re_mod % 3 - 1)
            mark[mark != 0] = 255
        img1 = self.memory.img_show.copy()
        img1[mark > 0] = 255
        self.memory.img_repair = Inpainting(img1, mark)

    # 朗读
    def tts(self):
        from gtts.tts import gTTS
        import pyglet
        if self.ui.textEdit.toPlainText().isspace() != True:
            try:
                tts = gTTS(text=self.ui.textEdit.toPlainText(), lang=self.var.img_language)
                filename = 'temp.mp3'
                tts.save(filename)
                music = pyglet.media.load(filename, streaming=False)
                music.play()
                time.sleep(music.duration)
                os.remove(filename)
            except:
                print('War:网络异常,TTS错误')

    # 切换翻译模式
    def change_translate_mod(self):
        change_translate_mod()
        if self.ui.textEdit.toPlainText().isspace() != True:
            self.ui.textEdit_2.setText(translate(self.ui.textEdit.toPlainText(), f'{self.var.word_language}', "auto"))

    # 参数保存
    def config_save(self, parameter, value):
        config = configparser.ConfigParser()
        config.read('config.ini')
        config.set('var', f'{parameter}', f'{value}')
        with open('./config.ini', 'w+') as config_file:
            config.write(config_file)

    # 参数读取
    def config_read(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.var.img_language = config.get('var', 'img_language')
        self.change_mod(self.var.img_language)

        self.var.word_language = config.get('var', 'word_language')
        self.change_out_language(self.var.word_language)

        self.var.word_mod = config.get('var', 'word_mod')
        if self.var.word_mod == 'auto':
            self.ui.pushButton_13.setText('定位:自动')
        else:
            self.ui.pushButton_13.setText('定位:手动')

        self.var.word_way = config.getint('var', 'word_way')
        if self.var.word_way == 1:
            self.ui.pushButton_2.setText('排列:垂直')
        else:
            self.ui.pushButton_2.setText('排列:横向')

        self.var.img_re_bool = config.getboolean('var', 'img_re_bool')
        if self.var.img_re_bool:
            self.ui.pushButton_8.setText('启用')
        else:
            self.ui.pushButton_8.setText('停用')

        self.var.img_re_mod = config.getint('var', 'img_re_mod')

        self.var.word_conf.font = config.get('var', 'font')
        self.ui.label_6.setText(self.var.word_conf.font)

        self.var.word_conf.color = config.get('var', 'color')
        self.var.word_conf.stroke_width = config.getint('var', 'stroke_width')
        self.var.word_conf.stroke_fill = config.get('var', 'stroke_fill')
        self.var.word_conf.line_spacing_factor = config.getfloat('var', 'line_spacing_factor')
        self.var.word_conf.letter_spacing_factor = config.getfloat('var', 'letter_spacing_factor')

# 字距设置窗口
class CharacterStyle(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.color = ''
        self.stroke_fill = ''
        self.ui = CharacterStyleDialog()
        self.setWindowIcon(QtGui.QIcon('img.png'))
        self.setWindowFlags(QtCore.Qt.WindowType.WindowCloseButtonHint)
        self.ui.setupUi(self)

        self.ui.lineEdit.setValidator(QtGui.QDoubleValidator())
        self.ui.lineEdit_2.setValidator(QtGui.QDoubleValidator())
        self.ui.lineEdit_3.setValidator(QtGui.QIntValidator())

        self.ui.pushButton.clicked.connect(self.ok)
        self.ui.pushButton_1.clicked.connect(self.change_word_colour)
        self.ui.pushButton_2.clicked.connect(self.close)
        self.ui.pushButton_3.clicked.connect(self.change_shadow_colour)
        self.re = [False, 0, 0, '', 0, '']

    def ok(self):
        self.re = [True, float(self.ui.lineEdit.text()), float(self.ui.lineEdit_2.text()), self.color,
                   int(self.ui.lineEdit_3.text()), self.stroke_fill]
        self.accept()

    def close(self):
        self.re = [False, 0, 0, '', 0, '']
        self.reject()

    def change_word_colour(self):
        r = colorchooser.askcolor(title='文字颜色')
        self.color = r[1]
        self.ui.pushButton_1.setStyleSheet(f'background-color: {r[1]};border-width:0px;border-radius:11px;')

    def change_shadow_colour(self):
        r = colorchooser.askcolor(title='阴影颜色')
        self.stroke_fill = r[1]
        self.ui.pushButton_3.setStyleSheet(f'background-color: {r[1]};border-width:0px;border-radius:11px;')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
