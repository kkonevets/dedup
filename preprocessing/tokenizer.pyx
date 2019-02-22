cimport cython

from cpython cimport array
import array
from cpython cimport bool

cdef:
    frozenset substitute = frozenset(u'ъ!"#$&\'()+-/:;<=>?@[\\]^_`{|}~')
    # '%*,.'


cpdef tokenize(s):
    cdef:
        unicode us = <unicode>s
        Py_UCS4 prev, last, c 
        unsigned int i = 0
        array.array out = array.array('u')
        unsigned long ordinal
        bool isnum = False , isdim = False

    for c in us:
        ordinal = ord(c)
        if not 32<=ordinal<=126 and not 208128<=ordinal<=209145:
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
            isnum = True
        else:
            if isnum and c != '%':
                out.append(' ')
            isnum = False

        if c == 'ё': c = 'е'
        if c == 'й': c = 'и'

        # separate numbers from strings
        # detect numbers like "0.6" or "33,5" 
        # detect dimentions 44x33 or 44*33

        out.append(c)
        last = c
        prev = c
        i += 1


    return out