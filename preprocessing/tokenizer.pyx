#!python
#cython: language_level=3

cimport cython

from cpython cimport array
import array
from cpython cimport bool
from icu import Transliterator

trans = Transliterator.createInstance('Latin-Cyrillic')

cdef:
    frozenset substitute = frozenset(u'ъ*!"#$&\'()+-/:;<=>?@[\\]^_`{|}~')
    # '%,.'
    dict unit_lookup = {
        'г': 'грамм', 'грам': 'грамм', 'гр': 'грамм', 'gr': 'грамм', 'g': 'грамм',
        'ml': 'мл', 'милл': 'мл', 'млитр': 'мл', 'млтр': 'мл',
        'ш': 'шт',
        'тон': 'тонна', 'тн': 'тонна', 'тонн': 'тонна', 'т': 'тонна',
        'л': 'литр', 'лит': 'литр',
        'kg': 'кг',
        'mm': 'мм', 'cm': 'см', 'м': 'метр', 'm': 'метр',
        'gb': 'гб', 'mb': 'мб', 
        '№': 'номер',
        'ват': 'ватт', 'вт': 'ватт', 'w': 'ватт', 'в': 'вольт', 'v': 'вольт',
        'а': 'ампер', 'a': 'ампер', 'hz': 'герц', 'гц': 'герц'}

    dict number_lookup = {
        '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
        '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь',
        '9': 'девять'
    }

cpdef tokenize(s):
    cdef:
        unicode ustring = <unicode>s
        Py_UCS4 prev, c 
        array.array out = array.array('u')
        unsigned int ordinal
        bool isnum = False
        bool islat = False, iscyr = False

    for c in ustring:
        ordinal = ord(c)
        islat = 32<=ordinal<=126
        iscyr = 1025<=ordinal<=1105

        if not islat and not iscyr:
            out.append(' ')
            continue

        c = c.lower()
        if c in substitute:
            out.append(' ')
            continue

        if c.isnumeric():
            if not isnum:
                out.append(' ')
            isnum = True
        elif isnum and (c == '.' or c == ','):
            c = '.'
            isnum = True
        else:
            if isnum and prev == '.':
                prev = ' '
                out[-1] = ' '
            elif isnum and c != '%':
                out.append(' ')
            elif c == '.' or c == ',':
                c = ' '
            elif prev == '%':
                out.append(' ')
            isnum = False

        if c == 'ё': c = 'е'
        if c == 'й': c = 'и'

        # detect dimentions 44x33 or 44*33

        out.append(c)
        prev = c

    cdef:
        list splited = out.tounicode().split()
        unsigned int lsp = len(splited)
        unicode w_prev, w_next
        list temp

    if lsp == 0:
        return ''

    temp = [splited[0]]
    if lsp > 1:
        for w_prev, w_next in zip(splited[:-1], splited[1:]):
            if w_prev.isnumeric():
                w_next = unit_lookup.get(w_next, w_next) 
            if len(w_next) > 1:
                temp.append(w_next)

    joined = ' '.join(temp)
    joined = trans.transliterate(joined)

    return joined
