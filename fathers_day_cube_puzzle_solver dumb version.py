# Father's Day Cube problem
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

import operator, itertools, time, collections

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

rel_faces = [right, back, left, front] = range(4)

# rel_to_abs is a dict to convert relative faces to absolute faces.
# The key of each dict entry is the absolute direction of the previous section
# going from the first cube to the last.
# For example, the first section has cubes 1, 2, 3.  The face on cube 1 that connects it to 2
# is the absolute direction used for the key.
# The value is a list of length len(rel_faces) with the conversion values.  To convert 'back'
# (which has value 1) to absolute faces when the previous section is going up, use
# rel_to_abs[up][back]

rel_to_abs = { up:  [east,north,west,south], north:[east,down,west,up],
               down:[east,south,west,north], south:[west,down,east,up],
               east:[south,down,north,up],    west:[north,down,south,up]
             }

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
initial_state_abs = [(3, south), (3, east), (3, south), (3, east),
                 (2, south), (2, east), (2, south), (3, east),
                 (3, south), (2, east), (2, south), (3, east),
                 (2, south), (3, east), (2, south), (2, east),
                 (3, south)]
# directions in initial_state_rel are relative except for the 0th item which needs to be absolute
initial_state_rel = [(3, south), (3, left), (3, right), (3, left),
                 (2, right), (2, left), (2, right), (3, left),
                 (3, right), (2, left), (2, right), (3, left),
                 (2, right), (3, left), (2, right), (2, left),
                 (3, right)]

solution_dims = (3, 3, 3) #specify the dimensions of the solution cube (in x, y, z, respectively)

def is_solution(state, solution_dimensions):
    '''Return True if given state is a valid solution
    Inputs:
        state: a list of tuples - one tuple for each section given the length
        of the section and the face of the previous section's last cube on
        which is it oriented.
        solution_dimensions: a 3-tuple given the dimensions of the cube
        which makes up a valid solution
    >>> is_solution([(3, 0), (3, 0), (3, 2), (3, 2), (2, 0), (2, 0), (2, 0), (3, 1), (3, 1), (2, 0), (2, 3), (3, 3), (2, 1), (3, 2), (2, 2)], (3,3,3))
    False
    >>> is_solution([(3, 0)], (0,3,0))
    ('is_right_size: True', [0, 3, 0], [(3, 0)])
    True
    '''
#    if is_right_size(state, solution_dimensions) and is_possible(state, solution_dimensions):
    if is_possible(state, solution_dimensions):
        return True
    else:
        return False

##def is_right_size(state, solution_dimensions):
##    '''Return True if given state fits into the size constraints without checking
##    whether it is possible (i.e. does not check if more than 1 cube is trying to
##    occupy the same space)
##    Inputs:
##        state: a list of tuples - one tuple for each section given the length
##        of the section and the face of the previous section's last cube on
##        which is it oriented.
##        solution_dimensions: a 3-tuple given the dimensions of the cube
##        which makes up a valid solution
##
##    >>> is_right_size([(3, 0), (3, 0), (3, 2), (3, 2), (2, 0), (2, 0), (2, 0), (3, 1), (3, 1), (2, 0), (2, 3), (3, 3), (2, 1), (3, 2), (2, 2)], (3,3,3))
##    False
##    >>> is_right_size([(3, 0)], (0,3,0))
##    ('is_right_size: True', [0, 3, 0], [(3, 0)])
##    True
##    >>> is_right_size([(3, 0), (3, 2), (3, 3)], (3,3,3))
##    ('is_right_size: True', [3, 3, 3], [(3, 0), (3, 2), (3, 3)])
##    True
##    >>> is_right_size([(3, 0), (3, 2), (3, 3)], (3,3,3))
##    ('is_right_size: True', [3, 3, 3], [(3, 0), (3, 2), (3, 3)])
##    True
##    '''
##    #print('In is_solution with state {}'.format(state)) #DEBUG
##    bb = [0]*len(solution_dimensions)
##    previous_section_heading = state[0][1]
##    bb[bb_input[south][0]] += state[0][0] #initialize first section in count
##    for i, section in enumerate(state[1:]):
##        length, rel_face = section
##        abs_face = rel_to_abs[previous_section_heading][rel_face]
##        bb_index, multiplier = bb_input[abs_face]
##        bb[bb_index] += length*multiplier
##        if debug: print('        {}: {}'.format(i, bb)) #DEBUG
##        if abs(bb[bb_index]) > solution_dimensions[bb_index]:
##            #print('   early exit at section {}, {}, {}'.format(i, bb, state)) #DEBUG
##            return False
##        previous_section_heading = abs_face
####    for i in range(len(solution_dimensions)):
####        if abs(bb[i]) != solution_dimensions[i]:
####            print('is_right_size: Within range but not exact: ',bb, state) #DEBUG
####            print_state(state) #DEBUG
####            return False
##    if debug: print('is_right_size: True', bb, state) #DEBUG
##    return True

def is_possible(state, solution_dimensions):
    '''Return True if the given state is physically possible, (i.e. it
    does not have multiple sections trying to occupy the same physical
    space'''
    # Assume that cube #1 of section #1 is at position (0, 0, 0).  Note
    # that this does not assume that cube #1 is at any particular position
    # in the cube. The cube could go into the negatives or positives, but
    # we need to check that the maximum range between high and low is not
    # more than the size of the solution_dimensions.
    # Keep a dict with keys as the coordinates (i.e. [0,0,0]) and the value
    # as # of blocks present.  If you go to put a block where one
    # already is, then return False.
    space = collections.defaultdict(int) #key = 3d coordinates, value = # blocks (int defaults 0)
    first_length, previous_section_heading = state[0]
    curr_location = [0,0,0]
    if debug: print(curr_location)
    max_xyz = [0,0,0]
    min_xyz = [0,0,0]
    space[tuple(curr_location)] = 1 # manually set the first block
    heading = previous_section_heading
    for block in range(1, first_length):
        axes, value = bb_input[heading]
        curr_location[axes] += value
        if debug: print(curr_location)
        max_xyz[axes] = max(max_xyz[axes], curr_location[axes])
        min_xyz[axes] = min(min_xyz[axes], curr_location[axes])
        space[tuple(curr_location)] = 1
    for i, section in enumerate(state[1:]):
        length, rel_heading = section
        heading = rel_to_abs[previous_section_heading][rel_heading]
        previous_section_heading = heading
        axes, value = bb_input[heading]
        for block in range(length-1): #first block of this section was last block of previous so skip it.
            curr_location[axes] += value
            if debug: print(curr_location)
            if space[tuple(curr_location)] > 0:
                if debug: print('NOT POSSIBLE',curr_location, space) #DEBUG
                return False
            else:
                space[tuple(curr_location)] = 1
            max_xyz[axes] = max(max_xyz[axes], curr_location[axes])
            min_xyz[axes] = min(min_xyz[axes], curr_location[axes])
            if max_xyz[axes] - min_xyz[axes] > solution_dimensions[axes]-1:
                if debug: print('TOO LARGE',max_xyz, min_xyz) #DEBUG
                return False
    if debug: print('POSSIBLE',max_xyz, min_xyz, space) #DEBUG
    return True

def print_state(state):
    '''Print the state in a human readable way in absolute coordinates'''
    previous_section_heading = south # always hold first section going south from 1 to 3
    abs_state = [(3, face_names[previous_section_heading])] #prepopulate with the first section
    num_state = [(3, previous_section_heading)]
    for s in state[1:]:
        abs_face = rel_to_abs[previous_section_heading][s[1]]
        abs_state.append( (s[0], face_names[abs_face]) )
        num_state.append( (s[0], abs_face) )
        previous_section_heading = abs_face
    print("{}\n{}\n{}\n***********".format(abs_state, num_state, state))

def fathers_day_puzzle(initial_state, solution_dimensions):
    '''The puzzle is 27 wooden cubes all connected to each other by an elastic
    string passing through holes drilled into the cubes.  The holes are drilled
    such that a string of cubes is never longer than 3 long before they angle
    the hole so the next section of cubes goes off at a 90 degree angle to the
    previous section.  Thus, you can rotate the sections relative to each other
    and can form a 3x3x3 cube.

    To simplify the problem, I consider it really just a collection of 17
    'sections' which are attached to each other at the last cube of the section
    such that two consecutive sections are always at a 90 degree angle to each
    other (i.e. the next section doesn't continue in the same direction).

    Since each section can be connected to the previous section on one of 4
    faces, there are 4**16 = 4.29 billion possible orientations (without
    excluding impossible orientations where multiple sections are rotated into
    the same space).  We will fix the orientation of the first section so
    the permutations are 4**15 = ~1B.

    The challenge is to write a program that finds a solution (defined as the
    "face" that each section is rotated to in order to connect to the previous
    section) that forms a 3x3x3 cube.
    Inputs:
        initial_state: a list of tuples - one tuple for each section given the length
        of the section and the face of the previous section's last cube on
        which is it oriented.
        solution_dimensions: a 3-tuple given the dimensions of the cube
        which makes up a valid solution
    '''

    initial_lengths = list(map(operator.itemgetter(0), initial_state))
    initial_faces = list(map(operator.itemgetter(1), initial_state))

    #print('Testing the print_state with initial_state:') #DEBUG
    #print_state(initial_state_rel) #DEBUG
    
    # the following statement is like 'itertools.product((up,down,east,west,north,south), (up,down,east,west,north,south), ...etc repeated 15 times)
    # Note, I subtract 2 from len(initial_state) because if I have 17 sections, I only have 16
    #       connections between them (-1) 

    permutations = itertools.product(*itertools.repeat(rel_faces, len(initial_state)-1)) 
    num_permutations = len(rel_faces)**(len(initial_state)-1)
    print('Total permutations = {}'.format(num_permutations))
    count = 0
    t0 = time.clock()
    found_sols = []
    try:
        #DAY TODO: prune the permutations by sending a custom exception in
        #          is_right_size when we have a beginning sequence which goes
        #          out of bounds.  Catch the exception here and modify permutations
        #          by using itertools.dropwhile(lambda x[:i] == (sd fsdf sdf )
        #          where 'i' and the (sdf sdf sfd) is from the initial segment
        #          that didn't work
        for p in permutations:
            count += 1
            this_state = list(zip(initial_lengths[1:], p))
            this_state.insert(0, initial_state[0]) #add back the first section (not in permutation)
            if is_solution(this_state, solution_dimensions):
                print("Found a possible solution:")
                print_state(this_state)
                found_sols.append(this_state)
    except KeyboardInterrupt:
        print('''KeyboardInterrupt: iteration count = {}, elapsed_time = {}h or {}m. Estimate completion in {}h'''.format(
            count, ((time.clock()-t0)/60)/60, ((time.clock()-t0)/60),
              (num_permutations/count)*((time.clock()-t0)/60)/60))
        raise KeyboardInterrupt
    finally:
        print(count)
        print('Found {} possible solutions'.format(len(found_sols)))
    return found_sols

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
    debug = False
    #doctest.testmod()
    debug = False

    is_solution([(3, 3), (3, 0), (3, 0), (3, 1), (2, 0), (2, 0), (2, 0), (3, 2), (3, 2), (2, 1), (2, 0), (3, 2), (2, 2), (3, 3), (2, 0), (2, 0), (3, 0)], [3,3,3])


    #if debug: print(timedcall(fathers_day_puzzle, [(3, south), (3, left), (3, right)], [3,3,3])) #DEBUG
    #if debug: print(timedcall(fathers_day_puzzle, [(3, south), (3, left), (3, right), (3, left)], [3,3,3])) #DEBUG
    #if debug: print(timedcall(fathers_day_puzzle, [(3, south), (3, left), (3, right), (3, left),(2, right), (2, left)], [3,3,3])) #DEBUG

    if not debug: print(timedcall(fathers_day_puzzle, initial_state_rel, solution_dims))










    
