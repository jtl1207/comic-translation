from .google import translate as gtranslate
from .deepL import DeepLTranslator as dtranslate

__all__ = ['translate', 'change_translate_mod']
mod = 1


def translate(text='', to_language="zh-CN", from_language="auto", in_mod=0):
    '''
    入口
    :param text: 输入
    :param to_language: 输入语言
    :param from_language: 输出语言
    :param in_mod: 强制模式
    :return: 输出
    '''
    if in_mod == 0:
        global mod
    else:
        mod = in_mod
    if text.replace(" ", "") == '':
        return 'War:文字识别异常,请手动输入'

    if from_language == 'ko' or to_language == 'ko' or mod == 3:
        if mod != 3:
            print('deepl暂不支持韩语')
        return gtranslate(text, to_language, from_language)
    else:
        if to_language == 'zh-CN':
            to_language = 'zh'
        if mod == 1:
            return dtranslate(translate_str=text, target_lang=to_language, translate_mode='word').translate()['result']
        elif mod == 2:
            return dtranslate(translate_str=text, target_lang=to_language, translate_mode='sentences').translate()[
                'result']


def change_translate_mod():
    '''
    切换模式
    '''
    global mod
    mod = mod + 1
    if mod == 4: mod -= 3
    if mod == 1:
        print('Info:使用deepl翻译模式1')
    elif mod == 2:
        print('Info:使用deepl翻译模式2')
    elif mod == 3:
        print('Info:使用Google翻译')
