cimport cython
from cpython cimport array
import array

cdef substitute = frozenset(u'~/-\[\]()|{}:^+')

cpdef tokenize(s):
    cdef:
        unicode us = <unicode>s
        Py_UCS4 prev, c
        unsigned int i = 0
        array.array out = array.array('u')

    for c in us:
        if c == u'ё':
            out.append(u'е')
        elif c in substitute:
            out.append(' ')
        else:
            out.append(c)
        
        i += 1
    
    return out