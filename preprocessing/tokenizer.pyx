cimport cython

cpdef tokenize(s):
    cdef:
        unicode us = <unicode>s
        Py_UCS4 c
    
    for c in us:
        c