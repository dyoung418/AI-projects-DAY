# -*- coding: utf-8 -*-
# Column Print 
#
import functools, operator, itertools, sys, time, copy, io

def decorator(d):
    """Make function d a decorator: d wraps a function fn.
    Note that update_wrapper just makes sure the docstring and args list
    in help(fn) point to the right place"""
    def _d(fn):
        return functools.update_wrapper(d(fn), fn)
    functools.update_wrapper(_d, d)
    return _d

@decorator
def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(args)
    return _f

@decorator
def trace(f):
    '''A decorator which prints a debugging trace every time the
    decorated function is called.  It handles recursion with
    indented print statements'''
    indent = '   '
    def _f(*args):
        signature = '%s(%s)' % (f.__name__, ', '.join(map(repr, args)))
        print '%s--> %s' % (trace.level*indent, signature)
        trace.level += 1
        try:
            # your code here
            result = f(*args)
            print '%s<-- %s == %s' % ((trace.level-1)*indent, 
                                      signature, result)
        finally:
            trace.level -= 1
            # your code here
        return f(*args)# your code here
    trace.level = 0
    return _f

def disabled(f): return f
#trace = disabled

def timedcall(fn, *args):
    "Call function with args; return the time in seconds and result."
    t0 = time.clock()
    result = fn(*args)
    t1 = time.clock()
    return t1-t0, result

def average(numbers):
    "Return the average (arithmetic mean) of a sequence of numbers."
    return sum(numbers) / float(len(numbers))

def timedcalls(n, fn, *args):
    """Call fn(*args) repeatedly: n times if n is an int, or up to
    n seconds if n is a float; return the min, avg, and max time"""
    if isinstance(n, int):
        times = [timedcall(fn, *args)[0] for _ in range(n)]
    else:
        times = []
        total = 0.0
        while total < n:
            t = timedcall(fn, *args)[0]
            total += t
            times.append(t)
    return min(times), average(times), max(times)




#############################################################################
###      Column Print Class (not implemented yet)
#############################################################################

class Column_Printer(object):
    '''Create a printing object, O, which takes multiple calls to O.add_text(text)
    and stores up the text until you call O.print() and all the added text
    is printed in columns (down for a page and then up to the next column)'''
    def __init__(self, num_cols, margins=3, pagewidth=80, pagelength=25):
        pass
    def add_text(self, text):
        pass
    def output(self):
        pass

#############################################################################
###      Column Print Functions
#############################################################################

def array_print_var_width(input_arrays):
    '''
    >>> rows =  [   ['a',           'b',            'c',    'd']
    ...         ,   ['aaaaaaaaaa',  'b',            'c',    'd']
    ...         ,   ['a',           'bbbbbbbbbb',   'c',    'd']
    ...         ]
    >>> array_print_var_width(rows)
    a           b           c  d
    aaaaaaaaaa  b           c  d
    a           bbbbbbbbbb  c  d
    '''
    widths = [max(map(len, col)) for col in zip(*input_arrays)]
    for row in input_arrays:
        print "  ".join((val.ljust(width) for val, width in zip(row, widths)))

def column_print(strings, columns=2, margin=" ", max_width=False, fixed_width=None, across_then_down=False):
    '''
    >>> column_print(['hello','my','name','is','Edwina Lawrence-Michaelopolis III','nice','to','meet','you'],columns=3)
    hello is                                to  
    my    Edwina Lawrence-Michaelopolis III meet
    name  nice                              you 
    >>> column_print(['hello','my','name','is','Edwina Lawrence-Michaelopolis III','nice','to','meet','you','today'],columns=3)
    hello Edwina Lawrence-Michaelopolis III you 
    my    nice                              today
    name  to                               
    is    meet                             
    >>> column_print(['hello','my','name','is','Edwina Lawrence-Michaelopolis III','nice','to','meet','you','today'],columns=3, across_then_down=True)
    hello my                                name
    is    Edwina Lawrence-Michaelopolis III nice
    to    meet                              you 
    today
    >>> column_print(['111\\n111','222\\n222','333\\n333','444\\n444'],columns=2, across_then_down=True)
    111     222
    111     222
    <BLANKLINE>
    333     444
    333     444
    <BLANKLINE>
    '''
    widths = [max(map(len, strings[j::columns])) for j in range(columns) ]
    if max_width:
        widths = [ max(widths) for i in range(len(widths))]
    if fixed_width:
        widths = [ fixed_width for i in range(len(widths))]
    if across_then_down:
        lines = [ strings[i:i+columns] for i in range(0,len(strings),columns)]
    else: # down then across
        num_rows = len(strings)//columns + (1 if len(strings)%columns != 0 else 0)
        lines = [ strings[i::num_rows] for i in range(num_rows)]
    for line in lines:
        #   If the strings in lines have embedded newlines ('\n'), then
        #     it need to be more complicated
        multiline_counts = map(len,[s.split('\n') for s in line])
        if max(multiline_counts) > 1:
            sublines = list(itertools.izip_longest(*[s.split('\n') for s in line], fillvalue=''))
            for sl in sublines:
                print(margin.join((val.ljust(width) for val, width in zip(sl, widths))))
            print('')
        else:
            # simple case: no string contains embedded newlines
            print(margin.join((val.ljust(width) for val, width in zip(line, widths))))

if __name__ == '__main__':

    global_debug = False
    if global_debug:
        import pdb
        pdb.set_trace()
    else:
        trace = disabled
    
    if True: #Doctest
        import doctest
        old_debug = global_debug
        global_debug = False
        print('Starting DocTest...')
        doctest.testmod()
        print('Done\n')
        global_debug = old_debug

    if True:
        f = io.open("pent_plus_sq_8x8_solutions.txt", mode='rt')
        solutions = f.read()
        f.close()
        s_list = solutions.split('#')
        print('Found {} solutions:\n'.format(len(s_list)))
        column_print(s_list, fixed_width=9, columns=8)
        
