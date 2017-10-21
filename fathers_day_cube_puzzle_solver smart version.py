# Father's Day Cube problem -- SMART VERSION
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

from __future__ import print_function
import operator, itertools, time, collections, functools

# faces represents the possible faces that the next section is connected to.  The
# last cube of the first section has 4 faces that can be connected to.  Keep
# these faces in an absolute reference system so that I can add up direction lengths
# easily later.
# Assume the table in front of you has x-axis going across (east, west), y-axis going toward
# and away from you (north,south) and z-axis lifting above and below the surface of the table
# (up,down).
# left and right are always from the perspective of a viewer that is looking at the 'origin'
# side of the cube (the numbered ones from my set). The origin side of the cube for the first
# section is always oriented facing up.

faces = [up, down, north, south, east, west] = range(6)
face_names = ['up', 'down', 'north', 'south', 'east', 'west' ]

successor_headings = {up:[north,east,west,south], down:[north,east,west,south],
                      north:[east,west,up,down], south:[east,west,up,down],
                      east:[north,south,up,down], west:[north,south,up,down]}

# To calculate the bounding box, assume that square #1 from the first section sits
# at 0,0,0 of the x,y,z axes with south in the positive direction for y, east in
# the positive direction for x and up in the positive direction for z
# Now bb_input gives the axis and multiplier to use for each direction in
# calculating the bounding box (to see if it fits in our solution_dims
bb_axes = [bb_x, bb_y, bb_z] = range(3)
bb_input = { up:(bb_z,1), down:(bb_z,-1), north:(bb_y,-1), south:(bb_y,1), east:(bb_x,1), west:(bb_x,-1) }

# State is given as a list of tuples.  Each tuple represents the next section
# and gives the length of the segment (this doesn't change within any given
# problem) and the face on which it is attached to the last cube of the
# previous section.
# Always hold the first section in a constant orientation
# where the first cube of the section is connected by it's "south" face to the
# rest of the section, thus the second section can be attached to one of
# 4 faces: [east (to the left as you look at this on the table since you need
#  to orient yourself facing south), down, west, up].
# In the solution, this first segment should always be considered to be on the
# bottom, with the section running south (as you go from 1 to 3 in the section)
# (this works if it is a 3x3x3
# cube and the first and second section are of length 3)
#
# In this initial state, my puzzle will be flat so it can lie on a table and
# will have all the numbers written on the cubes visible on the top surface.
initial_state = [(3, south), (3, east), (3, south), (3, east),
                 (2, south), (2, east), (2, south), (3, east),
                 (3, south), (2, east), (2, south), (3, east),
                 (2, south), (3, east), (2, south), (2, east),
                 (3, south)]

solution_dims = (3, 3, 3) #specify the dimensions of the solution cube (in x, y, z, respectively)

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
        signature = '{0}({1})'.format(f.__name__, ', '.join(map(repr, args)))
        print('{0}--> {1}'.format(trace.level*indent, signature))
        trace.level += 1
        try:
            # your code here
            result = f(*args)
            print('{0}<-- {1} == {2}' % ((trace.level-1)*indent, 
                                      signature, result))
        finally:
            trace.level -= 1
            # your code here
        return f(*args)# your code here
    trace.level = 0
    return _f

def disabled(f): return f
trace = disabled

@trace
def is_valid(node, solution_dimensions=solution_dims):
    '''Return True if the given state is physically possible up to the
    first partial_length segments, (i.e. it does not have multiple
    sections trying to occupy the same physical space and the entire
    thing fits in solution_dimensions).
    If is_valid -> True when partial_length == len(state), then the state
    is a goal state.'''
    # Assume that cube #1 of section #1 is at position (0, 0, 0).  Note
    # that this does not assume that cube #1 is at any particular position
    # in the finished 3x3x3 cube. The finished cube could go into the negatives or positives, but
    # we need to check that the maximum range between high and low is not
    # more than the size of the solution_dimensions.
    # Keep a dict with keys as the coordinates (i.e. [0,0,0]) and the value
    # as # of blocks present.  If you go to put a block where one
    # already is, then return False.
    state, partial_length = node
    if partial_length == 0:
        return True #first segment always fits in finished cube
    space = collections.defaultdict(int) #key = 3d coordinates, value = # blocks (int defaults 0)
    curr_location = [0,0,0]
    max_xyz = [0,0,0]
    min_xyz = [0,0,0]
    for i, section in enumerate(state[:partial_length+1]):
        length, heading = section
        previous_section_heading = heading
        axes, value = bb_input[heading]
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
            if max_xyz[axes] - min_xyz[axes] > solution_dimensions[axes]-1:
                if debug: print('TOO LARGE',max_xyz, min_xyz)
                return False # the total dimensions are too large
    if debug: print('POSSIBLE',max_xyz, min_xyz, space)
    return True


def print_state(state):
    '''Print the state in a human readable way in absolute coordinates'''
    printable_state = [ (s[0], face_names[s[1]]) for s in state]
    print('['),
    for l, d in printable_state:
        print("({},{}),".format(l, d)),
    print(']')

def successors(node):
    '''Return a list of successor nodes in the puzzle problem.
    A "node" is a (state, partial_length) tuple where partial_length
    is the zero-based index of the last element in the state to be considered
    (even though other placeholder segments are in the "state").'''
    state, partial_length = node
    _ , parent_heading = state[partial_length]
    sucessors = []
    new_partial_length = partial_length + 1
    if new_partial_length >= len(state):
        return [] # already at a leaf node, return empty set for sucessors
    segment_length, _ = state[new_partial_length]
    for heading in successor_headings[parent_heading]:
        state_copy = [ t for t in state ]
        state_copy[new_partial_length] = (segment_length, heading)
        sucessors.append( (state_copy, new_partial_length) )
    return sucessors

def depth_first_puzzle_search(state, successors, is_valid_partial, is_goal, all_sols=False):
    '''This implements a depth-first search through the possible
    configurations of the puzzle to find a solution.
    In the particular problem, the solution will only reside at the leaf nodes
    of the search tree, so we keep track of the length (depth) of the search tree
    and only test for goal state at the leafs.
    Also, we include a validity test so that we can prune branches that are
    invalid before the leaf nodes.
    Successor, is_valid_partial and is_goal are function arguments'''
    # Usually frontier includes state and path,
    #   but my state encapsulates the path, so omit path
    #   I also need to include partial_length in my frontier
    frontier = collections.deque([])
    frontier.append( (state, 0) ) # initialize partial_length to 0 (meaning 0th element of state is last)
    goal_length = len(state)-1
    solutions = []
    count_checks = 0
    while frontier:
        node = frontier.pop()
        if node[1] >= goal_length: #leaf node?
            if is_goal(node):
                solutions.append(node[0])
                if not all_sols:
                    print("Checked {} states".format(count_checks))
                    return solutions
            else:
                continue
        count_checks += 1
        if is_valid_partial(node):
            for n in successors(node):
                frontier.append(n)
    print("Checked {} states".format(count_checks))
    return solutions

@trace
def fathers_day_puzzle(in_state, solution_dimensions):
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
        answers = depth_first_puzzle_search(in_state, successors, is_valid, is_valid, all_sols=True)
        print('Found {} solutions'.format(len(answers)))
        for sol in answers:
            print_state(sol)
        return answers[0]
                
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
    import doctest
    debug = True
    #doctest.testmod()
    debug = False

    if debug: print(is_valid( ( [(3, south), (3, down), (3, north), (3, east),
                          (2, south), (2, west), (2, south), (3, up),
                          (3, north), (2, east), (2, down), (3, west),
                          (2, south), (3, east), (2, south), (2, up),
                          (3, east)], 16
                        ) ))
    debug = False
    if debug: print(timedcall(fathers_day_puzzle, [(3, south), (3, west), (3, east)], [3,3,3])) #DEBUG
    if not debug: print(timedcall(fathers_day_puzzle, initial_state, solution_dims))










    
