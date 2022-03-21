
import  time , eventlet,threading,os
from translate import translate


# text = translate("hello", "zh-CN", "auto", in_mod=3)
#
# print(text)


# 检测网络状态
def thread_net():
    t = time.time()
    # with eventlet.Timeout(10, False):
    text = translate("hello", "zh-CN", "auto", in_mod=3)
    if text != '你好':
        print('Error:网络异常,google翻译离线')
    else:
        print(f'google翻译:网络正常,ping:{(time.time() - t) * 1000:.0f}ms')

    t = time.time()
    with eventlet.Timeout(10, False):
        text = translate("hello", "zh-CN", "auto", in_mod=1)
        if text != '你好':
            print(f'Error:网络异常,deepl翻译离线')
        else:
            print(f'deepl翻译:网络正常,ping:{(time.time() - t) * 1000:.0f}ms')

    from gtts.tts import gTTS
    import pyglet
    try:
        with eventlet.Timeout(10, False):
            tts = gTTS(text='お兄ちゃん大好き', lang='ja')
        filename = 'temp.mp3'
        tts.save(filename)
        music = pyglet.media.load(filename, streaming=False)
        music.play()
        time.sleep(music.duration)
        os.remove(filename)
    except:
        print('Error:网络异常,语音离线')

thread_net = threading.Thread(target=thread_net())  # cuda
thread_net.setDaemon(True)
thread_net.start()