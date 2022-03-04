from PyQt6 import QtCore, QtGui, QtWidgets


# 提升table类
class TLabel(QtWidgets.QLabel):
    xy0 = [0, 0]
    xy1 = [0, 0]
    flag = False
    img_pos = [0, 0, 0, 0]
    flag_switch = False

    # 鼠标点击事件
    def mousePressEvent(self, event):
        self.flag = True
        self.img_pos = [0, 0, 0, 0]
        self.update()
        if self.flag and self.flag_switch:
            self.xy0[0] = event.position().x()
            self.xy0[1] = event.position().y()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag and self.flag_switch:
            self.xy1[0] = event.position().x()
            self.xy1[1] = event.position().y()
            if self.xy0[0] < self.xy1[0]:
                x = self.xy0[0]
            else:
                x = self.xy1[0]
            if self.xy0[1] < self.xy1[1]:
                y = self.xy0[1]
            else:
                y = self.xy1[1]
            w = abs(self.xy1[0] - self.xy0[0])
            if x + w > self.width():
                w = self.width() - x
            h = abs(self.xy1[1] - self.xy0[1])
            if y + h > self.height():
                h = self.height() - y
            if x < 0:
                x = 0
                w = self.xy0[0]
            if y < 0:
                y = 0
                h = self.xy0[1]
            self.img_pos = list(map(int, [x, y, w, h]))
            self.update()

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.flag_switch:
            rect = QtCore.QRect(self.img_pos[0], self.img_pos[1], self.img_pos[2], self.img_pos[3])
        else:
            rect = QtCore.QRect(0, 0, 0, 0)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 2, QtCore.Qt.PenStyle.SolidLine))
        painter.drawRect(rect)
