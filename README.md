# AI漫画翻译辅助工具
这是一个基于深度学习的漫画翻译辅助工具  
包含翻译，图像去字，自动嵌字功能。

目的是帮助非专业汉化人员完成更简单，快速的翻译任务。也可以用于加快专业翻译流程。  
ps：机翻有待加强，目前对横向文本支持较好，垂直文本非中文容易出现问题，希望更多的小众漫画能够得到汉化。
## 安装
开发者直接git  
注意paddle，pytorch，tensorflow2 需要官网下载，任意版本。    
注意30系显卡不支持cuda10.2。

普通用户这里[下载](https://github.com/jtl1207/comic-translation/releases "下载")   
显卡驱动版本过低将导致程序卡住，暂时不支持A卡。  
为了更好的显示错误原因，整合包保留了控制台。  
第一次打开会安装虚拟环境，需要7分钟，安装结束有报错没有关系(4个)。  
整合包确定正常使用后可以删除.\opt\packages文件夹(1.2Gb)
## 更新
如果出现小版本更新0.1内  
普通用户可以[下载](https://github.com/jtl1207/comic-translation/archive/refs/heads/main.zip )最新代码，放到整合包的resources文件夹内。
## 使用方法
1.提前排好图片顺序与名称  
2.打开软件（需要10秒）  
3.导入漫画，字体  
4.设置要翻译的语言  
5.手动选择区域，选择需要的功能  
6.手动修改译文，点击确定  
  
需要什么字体就导入什么字体比如空心字等  
使用相同颜色阴影可以加粗字体  
如果发生闪退等意外resources文件夹下save.jpg将会保存最新修改的图片文件  

想要尝试全自动翻译可以把整张图选中,然后点击自动翻译  
部分标点符号大小可能会渲染异常,更换排列方式有概率解决  
根据电脑配置运行速度不同，可能会出现卡顿  
  
### 报错处理:  
短时间卡顿是正常现象  
翻译超时与朗读超时请检测与translate.google.cn的网络连接 
字体无法渲染请切换字体  
  
### 环境要求:  
需要网络,使用Google翻译(中国)  
有显卡可显著加快图像处理速度(CUDA)  
cpu会非常慢  
需要驱动版本>=472.50  
整合包自带CUDA与cudnn不需要额外安装  

字体下载地址:  
[www.hellofont.cn](http://www.hellofont.cn "www.hellofont.cn")
## 效果
##### 界面展示(旧版)
[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/3.jpg "口语翻译是弱项")](http://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/3.jpg "口语翻译是弱项")  
##### 口语翻译有待加强
[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/2.png "口语翻译有待加强")](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/2.png "口语翻译有待加强")  
##### 其他功能
|[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/1.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/1.jpg)   |[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/1.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/1.jpg)   |
| ------------ | ------------ |
| [![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/7.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/7.jpg)  |  [![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/7.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/7.jpg) |
| [![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/14.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/14.jpg)  |[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/11.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/11.jpg)   |
| [![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/12.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/12.jpg)  |[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/12.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/12.jpg)   |
| [![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/13.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/in/13.jpg)  |[![](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/13.jpg)](https://github.com/jtl1207/comic-translation/blob/main/%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87/out/13.jpg)   |
## 目标
更多字体渲染方式（CG文字渲染）  
更多字体角度  
优化使用体验  
## 相关项目
[kha-white/manga-ocr](https://github.com/kha-white/manga-ocr "kha-white/manga-ocr")(垂直日文识别模型)  
[zyddnys](https://github.com/zyddnys/manga-image-translator)与[dmMaze](https://github.com/dmMaze/comic-text-detector)(文本检测模型)  
[KUR-creative/SickZil-Machine](https://github.com/KUR-creative/SickZil-Machine "KUR-creative/SickZil-Machine")(图像修复模型)
## 开源协议
BSD 3-Clause License  
感谢一切推广
