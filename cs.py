

import re

text = '啊啊啊 你\n好骚EHVEL HELLO啊!?66'

_RE_V_WORDS = re.compile(r"\w+|[!?|\"\'-]+", re.ASCII)
_RE_H_WORDS = re.compile(r"\+", re.ASCII)
ascii_words_range = ((x.start(), x.end())
                     for x in _RE_H_WORDS.finditer(text))
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
print(ret)