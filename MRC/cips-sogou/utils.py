# Tools
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            inside_code = ord(uchar)
        rstring += unichr(inside_code)
    return rstring


delCPunctuation = {
    u'\uff01': 1, u'\u3002': 1, u'\uffe5': 1, u'\u2026': 1, u'\uff09': 1, u'\uff08': 1,
                u'\u300b': 1, u'\u300a': 1, u'\uff0c': 1, u'\u3011': 1, u'\u3010': 1, u'\u2019': 1,
                u'\u2018': 1, u'\uff1b': 1, u'\uff1a': 1, u'\u201d': 1, u'\u201c': 1, u'\uff1f': 1
}
delStr = {
    '!': 1, '#': 1, '"': 1, '%': 1, '$': 1, "'": 1, '&': 1, ')': 1, '(': 1,
    ',': 1, ';': 1, ':': 1,
                '<': 1, '?': 1, '>': 1, '@': 1, '[': 1, ']': 1, '\\': 1, '_': 1, '^': 1,
                '`': 1, '{': 1, '}': 1, '|': 1, '~': 1
}


def drop_punctuation(utext):
    ustr = []
    for uchar in utext:
        if delCPunctuation.has_key(uchar) or delStr.has_key(uchar):
            continue
        ustr.append(uchar)
    return ''.join(ustr)


def alpha(uchar):
    n = ord(uchar)
    if n >= ord('a') and n <= ord('z') or n >= ord('A') and n <= ord('Z'):
        return True
    else:
        return False


def split_text(utext):
    '''
            input: unicode text
            output: tuple of token, alphabet and number will be togethor.
    '''
    utext = drop_punctuation(strQ2B(utext))
    result, token, last_stat = [], '', ''
    for i, uchar in enumerate(utext):
        cur_stat = ''
        if uchar.isdigit():
            cur_stat = 'NUM'
        elif alpha(uchar):
            cur_stat = 'ALPHA'
        elif uchar.isspace():
            cur_stat = 'SPACE'
        elif uchar == '-':
            cur_stat = last_stat
        elif uchar == '.':
            if last_stat == 'NUM' and i < len(utext) - 1 and utext[i + 1].isdigit():
                cur_stat = last_stat
            else:
                cur_stat = 'SPACE'
        if last_stat == '':
            if token:
                result.append(token)
            token, last_stat = '', cur_stat
        elif cur_stat == '' or cur_stat != last_stat:
            if token and last_stat != 'SPACE':
                result.append(token)
            token, last_stat = '', cur_stat
        token += uchar
    if token and last_stat != 'SPACE':
        result.append(token)
    return tuple(result)
# End