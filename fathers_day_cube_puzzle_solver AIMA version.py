# Father's Day Cube problem -- AIMA VERSION
#
# THIS VERSION USES THE OBJECTS AND ROUTINES FROM THE AIMA PYTHON FILES
#
# The puzzle is 27 wooden cubes all connected to each other by an elastic
# string passing through holes drilled into the cubes.  The holes are drilled
# such that a string of cubes is never longer than 3 long before they angle
# the hole so the next section of cubes goes off at a 90 degree angle to the
# previous section.  Thus, you can rotate the sections relative to each other
# and can form a 3x3x3 cube.
#
# To simplify the problem, I consider it really just a collection of 17
# 'sections' which are attached to each other at the last cube of the section
# such that two consecutive sections are always at a 90 degree angle to each
# other (i.e. the next section doesn't continue in the same direction).
#
# Since each section can be connected to the previous section on one of 4
# faces, there are 4**16 = 4.29 billion possible orientations (without
# excluding impossible orientations where multiple sections are rotated into
# the same space)
#
# The challenge is to write a program that finds a solution (defined as the
# "face" that each section is rotated to in order to connect to the previous
# section) that forms a 3x3x3 cube.

import search # AIMA CUSTOM MODULE
import time, collections, functools

######################################################################
######### Utility functions
######################################################################

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
trace = disabled

######################################################################
######### Problem Class
######################################################################

class Block_Puzzle(search.Problem):
    '''Wooden block puzzle of a string of 27 small wooden cubes connected
    by an interior string.  Every 2 or 3 blocks, the string turns at a
    90 degree angle (which can be rotated in one of 4 directions).
    The problem is to find a set of rotations between these segments
    which forms a larger (3x3x3) cube.'''

    segment_lengths = [3,3,3,3,2,2,2,3,3,2,2,3,2,3,2,2,3]
    faces = [up, down, north, south, east, west] = range(6)
    face_names = ['up', 'down', 'north', 'south', 'east', 'west' ]
    successor_headings = {up:[north,east,west,south], down:[north,east,west,south],
                          north:[east,west,up,down], south:[east,west,up,down],
                          east:[north,south,up,down], west:[north,south,up,down]}
    solution_dims = (3, 3, 3) #specify the dimensions of the solution cube (in x, y, z, respectively)
    bb_axes = [bb_x, bb_y, bb_z] = range(3)
    bb_input = { up:(bb_z,1), down:(bb_z,-1), north:(bb_y,-1), south:(bb_y,1), east:(bb_x,1), west:(bb_x,-1) }

    def actions(self, state):
        '''return the actions that can be executed in the
        given state.'''
        if not self.goal_test(state, partial=True):
            return [] #invalid partial state, don't return successor actions (prune tree)
        elif len(state) == 0:
            return [self.south] #start the first section going south
        return self.successor_headings[state[-1][1]] #the headings that can be used next

    def result(self, state, action):
        '''return the state after taking action on the given state'''
        if len(state) >= len(self.segment_lengths):
            return None # state is a leaf node -- cannot take further action
        return state + [ (self.segment_lengths[len(state)], action) ]

    def goal_test(self, state, partial=False):
        '''Return True if the given state is physically possible within
        a 3x3x3 grid.
        if partial==True, test partial solutions for validity.'''
        # Assume that cube #1 of section #1 is at position (0, 0, 0).  Note
        # that this does not assume that cube #1 is at any particular position
        # in the finished 3x3x3 cube. The finished cube could go into the negatives or positives, but
        # we need to check that the maximum range between high and low is not
        # more than the size of the solution_dimensions.
        # Keep a dict with keys as the coordinates (i.e. [0,0,0]) and the value
        # as # of blocks present.  If you go to put a block where one
        # already is, then return False.
        len_state = len(state)
        if len_state != len(self.segment_lengths) and not partial:
            return False #incomplete solution (partial)
        elif len_state <= 1 and partial:
            return True #first segment always fits in finished cube
        space = collections.defaultdict(int) #key = 3d coordinates, value = # blocks (int defaults 0)
        curr_location = [0,0,0]
        max_xyz = [0,0,0]
        min_xyz = [0,0,0]
        for i, section in enumerate(state):
            length, heading = section
            previous_section_heading = heading
            axes, value = self.bb_input[heading]
            for block in range(length-1): #first block of this section was last block of previous so skip it.
                curr_location[axes] += value
                if debug: print(curr_location)
                if space[tuple(curr_location)] > 0:
                    if debug: print('NOT POSSIBLE',curr_location, space)
                    return False # A block overlapped another block
                else:
                    space[tuple(curr_location)] = 1
                max_xyz[axes] = max(max_xyz[axes], curr_location[axes])
                min_xyz[axes] = min(min_xyz[axes], curr_location[axes])
                if debug: print('   max={}, min={}'.format(max_xyz, min_xyz))
                if max_xyz[axes] - min_xyz[axes] > self.solution_dims[axes]-1:
                    if debug: print('TOO LARGE',max_xyz, min_xyz)
                    return False # the total dimensions are too large
        if debug: print('POSSIBLE',max_xyz, min_xyz, space)
        return True

    def print_state(self, state):
        '''Print the state in a human readable way in absolute coordinates'''
        printable_state = [ (s[0], self.face_names[s[1]]) for s in state]
        print('['),
        for l, d in printable_state:
            print("({},{}),".format(l, d)),
        print(']')



def fathers_day_puzzle():
    '''Depth-first search through all possible states.  Check partial
    states (e.g. the first 4 segments, but not the later 10 segments)
    and if it fails, prune the rest of that tree and move on.
    partial_length tells us which segment we are searching on and recursion
    happens for every segment.
    The tree of solutions looks like this:
                                                   Seg1
               (right) r|              (back) b|              (left)    l|                  (front) f|
                      Seg2                  Seg2                      Seg2                      Seg2
               r|   b|   l|   f|            ...
               S3   S3   S3   S3
    '''
    t0 = time.clock()
    try:
        fathers_day_puzzle = Block_Puzzle([])
        answer = search.depth_first_tree_search(fathers_day_puzzle)
        fathers_day_puzzle.print_state(answer.state)
        return answer
                
    except KeyboardInterrupt:
        print('''KeyboardInterrupt: elapsed_time = {}h or {}m.'''.format(
            ((time.clock()-t0)/60)/60, ((time.clock()-t0)/60)))
        raise KeyboardInterrupt    

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

if __name__ == '__main__':
    #import doctest
    #doctest.testmod()
    debug = False

    print(timedcall(fathers_day_puzzle))










    
