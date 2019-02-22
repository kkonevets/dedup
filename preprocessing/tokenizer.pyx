#!python
#cython: language_level=3

cimport cython

from cpython cimport array
import array
from cpython cimport bool
from icu import Transliterator 

trans = Transliterator.createInstance('Latin-Cyrillic')

cdef:
    frozenset substitute = frozenset(u'ъ!"#$&\'()+-/:;<=>?@[\\]^_`{|}~')
    # '%*,.'

cpdef tokenize(s):
    cdef:
        unicode ustring = <unicode>s
        Py_UCS4 prev, c 
        unsigned int index = 0, slen = len(s)
        array.array out = array.array('u')
        unsigned int ordinal
        bool isnum = False, isdim = False
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
            if isnum and c != '%':
                out.append(' ')
            elif c == '.' or c == ',':
                c = ' '
            isnum = False

        if c == 'ё': c = 'е'
        if c == 'й': c = 'и'

        # detect dimentions 44x33 or 44*33
        # transliterate

        out.append(c)
        prev = c
        islat_prev = islat
        scyr_prev = iscyr
        index += 1

    return trans.transliterate(out.tounicode())
