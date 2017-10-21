# -*- coding: utf-8 -*-
# Pentomino puzzle 
#
# See this link: http://www.cs.brandeis.edu/~storer/JimPuzzles/PACK/Pentominoes/LINKS/PentominoesNivasch.pdf
# For an explanation that the Pentaminoes problem is equivalent to the Exact Cover problem which
# Donald Knuth shows an algorithm for in a paper titled "Dancing Links" out of Stanford University
# at this link: http://arxiv.org/pdf/cs/0011047v1.pdf
#
# Other good links:
#
# http://puzzler.sourceforge.net/     AND more specifically:   http://puzzler.sourceforge.net/docs/pentominoes.html
#
# http://www.mattbusche.org/blog/article/polycube/
#
# DAY TODO:
# 1. Get a successful run of Y45
#       * Problem, my algorithm generates too many solutions because it distinguishes pieces when it should not
#         (because they are all the same).  So an enclosure that has just 1 solution made of 8 identical shapes
#         should show just 1 solution, but my algorith will show 40320 = len(list(iteritems.permutations(range(8))))
# 
import functools, operator, itertools, sys, time, copy

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
###      EXACT COVER PROBLEM
#############################################################################

danny_ec = [  ['A', 'B', 'C', 'D'],
              [0,    0,   1,   1,],  #1
              [1,    1,   0,   1,],  #2
              [0,    0,   1,   0,],  #3
              [1,    0,   0,   0,],  #4
              [0,    0,   0,   1,],  #5
              [0,    1,   1,   0,],  #6
              [1,    0,   1,   1,],  #7
            ]

knuth_ec = [ ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
              [0,   0,   1,   0,   1,   1,   0],  #1
              [1,   0,   0,   1,   0,   0,   1],  #2
              [0,   1,   1,   0,   0,   1,   0],  #3
              [1,   0,   0,   1,   0,   0,   0],  #4
              [0,   1,   0,   0,   0,   0,   1],  #5
              [0,   0,   0,   1,   1,   0,   1],  #6
            ]

class StopAtOneException(Exception):
    pass
##    def __init__(self, value):
##        self.value = value
##    def __str__(self):
##        return repr(self.value)

class LLNode_ColEntry(object):
    def __init__(self, name, left, root=False, right=False):
        '''It is anticipated that columns are added to the linked list from left to right
        and that the column headers (this object) are created before any of the 1Entry objects'''
        self.N = name # Name of the column
        self.count = 0  # count of '1's in the column
        if root:
            self.L = self
            self.R = self
        else:
            self.L = left  # Left: The next column left  (or wrapped)
            self.R = self.L.R  # Right: The next column right  (or wrapped to root)
            self.L.R = self
            self.R.L = self
        self.U = self  # Up: The next node higher in the column (always wrapped to the bottom)
        self.D = self  # Down: The next node lower in the column (or wrapped up to the top)
    def __repr__(self):
        return 'LLNode_ColEntry: {}'.format(self.N)
    def increment_count(self):
        self.count += 1
    def decrement_count(self):
        self.count -= 1
    def remove_self(self): # we don't remove self from the column; only from the row of columns yet to be assigned
        self.L.R = self.R
        self.R.L = self.L
    def reinsert_self(self):
        self.R.L = self
        self.L.R = self
    def cover_self(self):
        '''Remove this column from the column header list and remove all rows in this
        column's own list from other columns they are in.'''
        pass
        
        
class LLNode_1Entry(object):
    def __init__(self, column, up, left, right=None, name=None):
        '''It is anticipated that the 1Entry objects (this object) are created from topmost to bottommost
        and from left to right'''
        self.U = up  # Up: The next node higher in the column (or wrapped to the bottom)
        self.D = self.U.D  # Down: The next node lower in the column (or wrapped up to the top)
        self.U.D = self
        self.D.U = self
        if left==None:
            self.L = self   # Left: The next node left in the row (or wrapped)
            self.R = self   # Right: The next node right in the row (or wrapped)
        else:
            self.L = left
            self.R = self.L.R
            self.L.R = self
            self.R.L = self
        self.name = name
        self.C = column # Column Header
        self.C.increment_count()
    def __repr__(self):
        return 'LLNode_1Entry: row {0} col {1}'.format(self.name, self.C.N)
    def remove_self(self):
        #self.R.L = self.L  #The DLX algorithm doesn't need/want me to remove the node from the row linked list -- only the column
        #self.L.R = self.R
        self.D.U = self.U
        self.U.D = self.D
        self.C.decrement_count()
    # Note that if multiple nodes in a row or column are "remove_self'd", then those nodes need to
    #   have "reinsert_self" called in the opposite order.  For example, if I
    #   call node1.remove_self(), node2.remove_self(), node3.remove_self
    #   then to undo, I must call reinsert_self() on node3 first, then node2, then node1
    def reinsert_self(self): 
        self.C.increment_count()
        self.U.D = self   
        self.D.U = self
        #self.L.R = self  #The DLX algorithm doesn't need/want me to remove the node from the row linked list -- only the column
        #self.R.L = self

class Exact_Cover(object):
    '''Defines an exact cover problem and provides a Solve_DLX method
    to give the possible solutions.
    '''

    def __init__(self, array, stopatfirstsolution=False, verbose=False, debug=False):
        '''Given an array of 0 and 1s, the exact cover problem is to find a subset of rows
        (or all such subset of rows)
        which between them have every column covered with a '1' in exactly one row.  Every column
        is covered and no column is covered more than once.

        The given array is a list of lists.  The internal lists are rows in the array and contain
        a set of integers (0 or 1).  The first row is a column header.  For example:
        [ ['A', 'B', 'C', 'D'],
          [0,    0,   1,   1,],  #1
          [1,    1,   0,   1,],  #2
          [0,    0,   1,   0,],  #3
          [1,    0,   0,   0,],  #4
          [0,    0,   0,   1,],  #5
          [0,    1,   1,   0,],  #6
          [1,    0,   1,   1,],  #7
        ]

        In the exact cover problem given by the array above, there are 2 solutions:  one with rows 2 and 3, the
        other with rows 4, 5 and 6.  Row 7 doesn't participate in any possible solutions.

        '''
        self.verbose = verbose
        self.debug = debug
        if debug: self.verbose = True
        self.stopatfirstsolution = stopatfirstsolution
        self.input_array = array
        self.input_num_columns = len(self.input_array[0])
        self.input_num_rows = len(self.input_array)-1        
        self.root = self.initialize_linked_lists()
        self.solutions = []
        if self.debug: print('Done initializing Exact_Cover in __init__')

    def initialize_linked_lists(self):
        '''Create the doubly-linked-list which is used to represent the sparse matrix of self.input_array.
        Return the root node of this list.

        Here is the format of the doubly-linked-list:


                   |                    |                      |
                   V                    V                      V             
        root --> column1_header --> column2_header --> column3_header --> etc.
                   |                    |                      |
                   V                    |                      V
             ---'1'entry ---------------------------------> '1' entry -----> (wrap around)
                   |                    |                      |
                   |                    V                      V
             ------|-------------->'1' entry -----------> '1' entry -----> (wrap around)
                   |                    |                      |
                   V                    V                      V
                (wrap to col)        (wrap)                  (wrap)

        '''
        # Root
        self.root = LLNode_ColEntry('ROOT', None, root=True)
        # Column Headers
        previous = self.root
        for column_name in self.input_array[0]:
            previous = LLNode_ColEntry(column_name, previous)
        # 1Entry Nodes
        for index, row in enumerate(self.input_array[1:]):
            current_column = self.root.R     # 'carriage return' back to the first column on each iteration
            previous_row_1entry = None
            previous_col_1entry = current_column.U #this will wrap to bottom-most column entry (or column header at first)
            for entry in row:
                if entry == 1:
                    node = LLNode_1Entry(current_column, previous_col_1entry, previous_row_1entry, name=index+1)
                    previous_row_1entry = node
                current_column = current_column.R
                previous_col_1entry = current_column.U
        if self.debug: self.display_ll()
        return self.root
                
    def solve_DLX(self):
        '''Solve the Exact Cover problem using Knuth's dancing links X algorithm (DLX)

        General algorithm (taken from Knuth paper)
        If A is empty, the problem is solved; terminate successfully
        Otherwise, choose a column, c (deterministically -- e.g. column with least 1s)
        Choose a row, r, such that A[r,c] = 1 (nondeterministically) #if no such row exists, exit unsuccessfully
        Include row r in the partial solution.
        For each j such that A[r,j] = 1, # for all columns where row r has a 1...(including r itself?)
            delete the entire column j from matrix A; #
            for each i such that A[i,j] = 1, #for all rows i that have a 1 in each of the columns j that were deleted
                delete row i from matrix A.
        Repeat this algorithm recursively on the reduced matrix A

        >>> ec = Exact_Cover(danny_ec)
        >>> ec.solve_DLX()
        >>> len(ec.solutions)
        2
        '''

        def _search(k, current_solution):
            if self.root.R == self.root:
                # Success! pull solution from the algorithm and return it or add it to the list of solutions found
                self.solutions.append(tuple(current_solution))
                #self.display_solution(self.solutions[-1]); print('') #DEBUG!!
                if self.verbose: self.display_solution(current_solution)
                if self.stopatfirstsolution:
                    raise StopAtOneException
                current_solution = [None for _ in range(self.input_num_rows)]
                return
            # choose the next column (the one with minimum column size to minimize branching)
            curr_col = self.min_size_col()
            if self.debug: print('\n'+' '*(k*4)+'MAJOR COLUMN: {}'.format(curr_col.N))
            # Cover curr_col
            self.cover_column(curr_col, k)
            # for each row in curr_col with a 1
            curr_row = c = curr_col.D
            while curr_row != curr_col:
                if self.debug: print('\n'+' '*(k*4)+'minor row: {}'.format(curr_row.name))
                # add this row to the solution
                current_solution[k] = curr_row
                # for each neighboring column with entry in this row
                neighbor_row = curr_row.R
                while neighbor_row != curr_row:
                    # cover that neighboring column
                    self.cover_column(neighbor_row.C, k)
                    neighbor_row = neighbor_row.R
                # recursively call search() on the reduced linked list array
                if self.verbose: print('\n'+' '*(k*4)+'Before Recursion: curr_sol = {}\n'.format(['{}{}'.format(n.C.N, n.name) for n in current_solution if n]))
                _search(k+1, current_solution)
                r = current_solution[k]
                c = r.C
                neighbor_row = r.L
                while neighbor_row != r:
                    self.uncover_column(neighbor_row.C, k)
                    neighbor_row = neighbor_row.L
                curr_row = curr_row.D
            # Note that if we never enter the while loop above, that is a signal
            #   of failure of the partial solution since there was no row to cover curr_col.
            #   So, the code below is either run when a solution has failed, or at the end
            #   when the recursion is unwinding.  (This whole algorithm leaves the double-linked
            #   list as it was in the beginning by the time it exits).
            self.uncover_column(c, k)
            return

        
        current_solution = [None for _ in range(self.input_num_rows)]

        try:
            _search(0, current_solution)
        except StopAtOneException:
            print('Stopping at first solution')
        finally:
            return
                
                
    def cover_column(self, curr_col, k=None):
        '''Knuth: "removes curr_col from the header list and removes all rows
        in curr_col’s own list from the other column lists they are in.

        Note that this method does nothing (except curr_col.remove_self())
        if there are no rows in curr_col(i.e. if curr_col.D == curr_col)  That is OK
        because in that scenario solve_DLX will immediately uncover curr_col

        Also note that covering a column does *not* unlink the 1nodes from the
        column being covered -- it only unlinks 1nodes from uncovered columns
        where the 1node is in the same row as a 1node in the covered column.
        '''
        if self.verbose: print(' '*(k*4)+'L{0}: covering {1}'.format(k, curr_col.N))
        # remove curr_col from column header list
        curr_col.remove_self()
        # for each row in this column with a 1
        row_with_1_in_currcol = curr_col.D
        while row_with_1_in_currcol != curr_col:
            # for each neighboring column with entry in this row
            neighbor_row_1 = row_with_1_in_currcol.R
            while neighbor_row_1 != row_with_1_in_currcol:
                # remove the 1 entry in this column/row
                neighbor_row_1.remove_self()
                neighbor_row_1 = neighbor_row_1.R
            row_with_1_in_currcol = row_with_1_in_currcol.D
        if self.debug: self.display_ll(k)


    def uncover_column(self, curr_col, k=None):
        '''Reinserts a previously covered column'''
        if self.verbose: print(' '*(k*4)+'L{0}: UN-covering {1}'.format(k, curr_col.N))
        next_node_bottoms_up = curr_col.U
        while next_node_bottoms_up != curr_col:
            prev_node_rightleft = next_node_bottoms_up.L
            while prev_node_rightleft != next_node_bottoms_up:
                prev_node_rightleft.reinsert_self()
                # move to next while condition
                prev_node_rightleft = prev_node_rightleft.L
            # move to next while condition
            next_node_bottoms_up = next_node_bottoms_up.U
        curr_col.reinsert_self() # re-add the column in the header linked list
        if self.debug: self.display_ll(k)
            
    def min_size_col(self):
        s = sys.maxint # i.e. infinity
        col = self.root.R
        while col != self.root:
            if col.count < s:
                s = col.count
                min_col = col
            col = col.R
        return min_col

    def display_solution(self, solution):
        solution_text = []
        for O_entry in solution:
            if not O_entry:
                break
            row = []
            first_col = O_entry
            row.append('#{}: '.format(first_col.name))
            row.append(first_col.C.N)
            col = first_col.R
            while col != first_col:
                row.append(col.C.N)
                col = col.R
            solution_text.append(sorted(row))
        print('SOLUTION: ')
        solution_text_str = [col_name if col_name is str else str(col_name) for col_name in solution_text]        
        print('\n'.join([ ''.join(row) for row in sorted(solution_text_str)]))
        print('-------')
            
    def display_ll(self, tabs=None):
        '''Print the linked list for debug purposes'''
        array = []
        col_names = []
        covered_nodes = []
        if tabs: indent = ' '*(tabs*4)
        else: indent = ''
        # First gather all column names and order
        col = self.root.R
        while col != self.root:
            col_names.append(col.N)
            col = col.R
        col_ind = {c:col_names.index(c) for c in col_names}
        row_indices = []
        # Go to every Node and add it's row if it hasn't already been covered in a previous row
        col = self.root.R
        while col != self.root:
            node = col.D
            while node != col:
                row_entry = ['0' for c in col_names]
                if node not in covered_nodes:
                    covered_nodes.append(node)
                    row_indices.append(node.name)
                    row_entry[col_ind[node.C.N]] = '1'
                    row_neighbor_node = node.R
                    while row_neighbor_node != node:
                        covered_nodes.append(row_neighbor_node)
                        row_entry[col_ind[row_neighbor_node.C.N]] = '1'
                        row_neighbor_node = row_neighbor_node.R
                    array.append(row_entry)
                node = node.D
            col = col.R
        # Reorder the rows in their original order (which I get from the row.name attribute)
        decorated = [ (row_indices[i], r) for i, r in enumerate(array) ] # decorate the array with indices for sort
        decorated.sort()
        #array = [r for i, r in decorated] #undecorate
        #array.insert(0, col_names) # insert column names
        col_names_str = [col_name if col_name is str else str(col_name) for col_name in col_names]
        print(indent+'    ' + '  '.join(col_names_str)) # print column names (with space at left to align after row #)
        #print('\n'.join(['#'+str(i)+': '+'  '.join(r) for i, r in decorated]))
        for i, r in decorated:
            print(indent+'#'+str(i)+': '+'  '.join(r))

        
    def traverse(self, start_node, direction='R'):
        '''Iterator that give the next node in the linked
        list in the direction given.  direction can be
        'R', 'L', 'U' and 'D'.'''
        pass
            
            


#############################################################################
###      PENTOMINOES PROBLEM
#############################################################################

rectangle10by6 = ['1111111111' for i in range(6)] # has 2339 solutions

square8by8withhole = ['11111111',
                       '11111111',
                       '11111111',
                       '111XX111',
                       '111XX111',
                       '11111111',
                       '11111111',
                       '11111111'] # has 65 solutions

square8by8 = ['11111111' for i in range(8)] # for use with Pent_plus_square

rectangle3x20 = ['11111111111111111111' for i in range(3)] # has 2 solutions

big_F = ['XXX111111XXXXXX',
         'XXX111111XXXXXX',
         'XXX111111XXXXXX',
         '111111XXXXXXXXX',
         '111111XXXXXXXXX',
         '111111XXXXXXXXX',
         'XXX111XXXXXXXXX',
         'XXX111XXXXXXXXX',
         'XXX111XXXXXXXXX',
         ] # can be made with a subset of 9 of the pieces (not counting the 'F' and 2 others)

def make_bigshape_enclosure(bigshape):
    '''
    >>> F = make_bigshape_enclosure('F')
    >>> F == big_F
    True
    >>> F
    ['XXX111111XXXXXX', 'XXX111111XXXXXX', 'XXX111111XXXXXX', '111111XXXXXXXXX', '111111XXXXXXXXX', '111111XXXXXXXXX', 'XXX111XXXXXXXXX', 'XXX111XXXXXXXXX', 'XXX111XXXXXXXXX']
    '''
    enclosure = []
    coords = pentominoes[bigshape]['c']
    for row in range(5):
        rowstring = ''
        for col in range(5):
            if (col,row) in coords:
                rowstring += '111'
            else:
                rowstring += 'XXX'
        if rowstring != 'XXXXXXXXXXXXXXX':
            enclosure.append(rowstring)
            enclosure.append(rowstring)
            enclosure.append(rowstring)
    return enclosure   
                  
square15x15 = ['111111111111111' for i in range(15)]  #for use with the 45Y problem, has 212 solutions
                
square15x15_predef = [  'X1111111111XXXX',        #Has 4 spots predefined for Y41
                        'XX11111111111X1',
                        'X11111111111111',
                        'X11111111111111',
                        '111111111111111',
                        '111111111111111',
                        '111111111111111',
                        '111111111111111',
                        '111111111111111',
                        '111111111111111',
                        '111111111111111',
                        '11111111111111X',
                        '11111111111111X',
                        '1X11111111111XX',
                        'XXXX1111111111X',] # has 212 solutions

eightYs = [ 'XXX1XXXX',        # Easy solution for 8 Ys
            'X111111X',
            'X111111X',
            'X1111111',
            '1111111X',
            'X111111X',
            'X111111X',
            'XXXX1XXX',]

pentominoes = {
            'N':{'s':((1,0,0,0,0), # helps me visually see the shape, but not otherwise used
                      (1,1,0,0,0),
                      (0,1,0,0,0),
                      (0,1,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(0,1),(1,1),(1,2),(1,3)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'F':{'s':((0,1,1,0,0),
                      (1,1,0,0,0),
                      (0,1,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((1,0),(2,0),(0,1),(1,1),(1,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'Y':{'s':((0,0,1,0,0),
                      (1,1,1,1,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((2,0),(0,1),(1,1),(2,1),(3,1)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'T':{'s':((1,1,1,0,0),
                      (0,1,0,0,0),
                      (0,1,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(1,0),(2,0),(1,1),(1,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':((1,0),(1,1),(1,2)),   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'U':{'s':((1,0,1,0,0),
                      (1,1,1,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(2,0),(0,1),(1,1),(2,1)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':((1,1)),   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'W':{'s':((1,0,0,0,0),
                      (1,1,0,0,0),
                      (0,1,1,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(0,1),(1,1),(1,2),(2,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'L':{'s':((1,0,0,0,0),
                      (1,0,0,0,0),
                      (1,0,0,0,0),
                      (1,1,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(0,1),(0,2),(0,3),(1,3)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'I':{'s':((1,0,0,0,0),
                      (1,0,0,0,0),
                      (1,0,0,0,0),
                      (1,0,0,0,0),
                      (1,0,0,0,0)),
                 'c':((0,0),(0,1),(0,2),(0,3),(0,4)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':((0,0),(0,1),(0,2),(0,3),(0,4)),   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'Z':{'s':((1,1,0,0,0),
                      (0,1,0,0,0),
                      (0,1,1,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(1,0),(1,1),(1,2),(2,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'X':{'s':((0,1,0,0,0),
                      (1,1,1,0,0),
                      (0,1,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((1,0),(0,1),(1,1),(2,1),(1,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':((1,0),(0,1),(1,1)),  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':((1,1)),   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':((1,1)), #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'V':{'s':((1,0,0,0,0),
                      (1,0,0,0,0),
                      (1,1,1,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(0,1),(0,2),(1,2),(2,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            'P':{'s':((1,1,0,0,0),
                      (1,1,0,0,0),
                      (1,0,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((0,0),(1,0),(0,1),(1,1),(0,2)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 },
            }

pent_plus_square = copy.deepcopy(pentominoes)
pent_plus_square['S'] = {'s':((1,1,0,0,0),
                              (1,1,0,0,0),
                              (0,0,0,0,0),
                              (0,0,0,0,0),
                              (0,0,0,0,0)),
                 'c':((0,0),(1,0),(0,1),(1,1)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':1
                 }

Y45 = {
            'Y':{'s':((0,0,1,0,0),
                      (1,1,1,1,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((2,0),(0,1),(1,1),(2,1),(3,1)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':45,
                 'labels':[chr(ord('A')+i) for i in range(45)],
                 },
        }
        
Y41 = { #For use in 15x15 enclosures which pre-define 4 of the locations of Ys
            'Y':{'s':((0,0,1,0,0),
                      (1,1,1,1,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((2,0),(0,1),(1,1),(2,1),(3,1)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':41,
                 'labels':[chr(ord('A')+i) for i in range(41)],
                 },
        }

Y8 = { #For use in 15x15 enclosures which pre-define 4 of the locations of Ys
            'Y':{'s':((0,0,1,0,0),
                      (1,1,1,1,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0),
                      (0,0,0,0,0)),
                 'c':((2,0),(0,1),(1,1),(2,1),(3,1)), #coordinates of 1s in this rotation/flip
                 'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
                 'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
                 'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
                 'n':8,
                 'labels':[chr(ord('A')+i) for i in range(8)],
                 },
        }


class Pentomino(Exact_Cover):
    '''Implement the pentomino problem as a specific example of an exact cover problem.

    >>> pent = Pentomino(square8by8withhole, pentominoes)
    >>> len(pent.input_array)
    1569

    "comment out" the following doctest because it takes ~45 seconds.
    > pent3x20 = Pentomino(rectangle3x20, pentominoes)
    > pent3x20.solve_DLX()
    > len(pent3x20.solutions)
    2

    '''
    
    def __init__(self, enclosure, shape_defs, stopatfirstsolution=False, verbose=False, debug=False, 
                placements_to_exclude=None, skip_sizecheck=False):
        ''' Create a Pentomino problem object which translates the problem to an Exact Cover problem and provides
        Knuth's DLX (Dancing Links X) algorithm to solve it.
        Inputs:
        enclosure = a dict: keys=shape_names values=dict ->{'c':tuple of coords, 'limited_centers':[]
                                                            'noflip_centers':[], 'norotate_centers':[],
                                                            'n':1 (or higher number if multiple copies of this shape)
                                                            'labels': [list of label names provided if n>1] }
        enclosure = a list of strings.  All strings must be the same length and have either '1' for valid placement cell or 'X' for invalid placement cell
        placements_to_exclude = (optional) a dict: key=shape_name, value=list of coordinates in the enclosure coordinate space which should not be considered.
        '''
        ec_array = self.create_ExactCover_grid(enclosure, shape_defs, placements_to_exclude, skip_sizecheck)
        if debug: print('Done creating ExactCover grid.\n{}\n\nCalling Exact_Cover.__init__()'.format(ec_array))
        Exact_Cover.__init__(self, ec_array, stopatfirstsolution=stopatfirstsolution, verbose=verbose, debug=debug)
        
    def init_symmetry_flags(self):
        # Identify any symmetries and set flags
        self.flip_symmetry = self.rotate_symmetry = False
        if self.flip_enclosure_text(self.enclosure_text) == self.enclosure_text:
            self.flip_symmetry = True
        if (self.rotate_enclosure_text(self.enclosure_text) == self.enclosure_text) or (self.rotate_enclosure_text_180(self.enclosure_text) == self.enclosure_text):
            self.rotate_symmetry = True

    def create_ExactCover_grid(self, enclosure, shapes, placements_to_exclude=None, skip_sizecheck=False):
        self.enclosure = enclosure
        self.enclosure_text = '\n'.join(enclosure)
        self.init_symmetry_flags()
        if not skip_sizecheck:
            assert all( (len(enclosure[i])==len(enclosure[i+1]) for i in range(len(enclosure)-1)) ) #all lines of enclosure must be equal length
        self.shapes = shapes
        self.pentomino_orientations = self.enumerate_pentomino_orientations()
        # Count number of cells in the given shapes (normal pentaminoes have 60 cells)
        self.num_cells = sum([len(self.shapes[key]['c'])*self.shapes[key]['n'] for key in self.shapes])
        # In normal Pentominos, our exact cover array will have 72 columns: 12 for the
        # pentomino shapes and 60 for each of the cells in the enclosure shape that
        # we are fitting them into.
        #
        # We will create one row in the array for each of the ways that a given pentomino
        # (marked by a 1 in that pentomino's column) can be validly fit into the enclosure
        # (with a 1 in the column for each cell that the pentomino occupies).
        # Create Column headers for the Exact Cover grid.
        self.pentomino_col_names = list(self.pentomino_orientations.keys())
        self.cell_col_names = []
        for y, row in enumerate(self.enclosure):
            for x, value in enumerate(self.enclosure[y]):
                if value == '1':
                    self.cell_col_names.append( (x,y) ) #use the x,y coordinates for the cell column names
        if not skip_sizecheck:
            assert len(self.cell_col_names) == self.num_cells # num cells in the shapes must match num cells in enclosure
        ec_array = []
        ec_array.append(self.pentomino_col_names + self.cell_col_names) # Add the column header row
        # Create 1 Row for each possible piece placement in the enclosure for every piece.
        for pentomino_name in self.pentomino_col_names:
            i = self.pentomino_col_names.index(pentomino_name)
            # the first X (i.e. 12) columns of every row will be entries for the shapes (all zero except for one 1).  Generate that here for this piece and use it for every row for this piece
            row_pent_portion = list(itertools.chain((0 for x in range(i)), (1 for x in range(1)), (0 for x in range(len(self.pentomino_col_names)-1-i))))
            valid_placements = self.list_all_placements(pentomino_name)
            if placements_to_exclude:
                if pentomino_name in placements_to_exclude:
                    valid_placements = list( set(valid_placements) - set(placements_to_exclude[pentomino_name]) )
            for placement in valid_placements:
                row_cell_portion = [0 for x in range(self.num_cells)] #initialize to 0's
                for coord in placement:
                    row_cell_portion[self.cell_col_names.index(coord)] = 1
                ec_array.append(row_pent_portion + row_cell_portion) # add row for every valid placement of every pentomino
        return ec_array

    def solve_DLX(self):
        '''Wrapper around solve_DLX to put the solutions into a Pentomino-friendly form
        and de-duplicate symmetrically identical solutions'''
        Exact_Cover.solve_DLX(self)
        if self.debug: print('Done with EC Solution -- translating for Pentominos')
        self.solutions_text = []
        for s in self.solutions:
            s_text = self.placement_text(self.create_solution_dict(s))
            duplicate = False
            if self.flip_symmetry:
                flipped = self.flip_enclosure_text(s_text)
                if flipped in self.solutions_text:
                    duplicate = True
                if self.rotate_symmetry: #ie. both flip and rotate symmetry
                    if self.rotate_enclosure_text(flipped) in self.solutions_text:
                        duplicate = True
                    if self.rotate_enclosure_text_180(flipped) in self.solutions_text:
                        duplicate = True
                    if self.rotate_enclosure_text(self.rotate_enclosure_text_180(flipped)) in self.solutions_text:
                        duplicate = True
            if self.rotate_symmetry:
                if self.rotate_enclosure_text(s_text) in self.solutions_text:
                    duplicate = True
                if self.rotate_enclosure_text_180(s_text) in self.solutions_text:
                    duplicate = True
                if self.rotate_enclosure_text(self.rotate_enclosure_text_180(s_text)) in self.solutions_text:
                    duplicate = True
            if not duplicate:
                self.solutions_text.append(s_text)

    def solve_with_subset(self, n, enclosure=None, shapes=None, placements_to_exclude=None):
        '''Solve the enclosure with a subset (count=n) of the shapes'''
        if enclosure: self.enclosure = enclosure
        self.master_shapes = self.shapes
        if shapes: self.shapes = shapes
        shape_names = self.shapes.keys()
        total_solutions = []
        for subset_names in itertools.combinations(shape_names, n):
            subset = {k:v for k,v in self.master_shapes.iteritems() if k in subset_names}
            self.__init__(self.enclosure, subset, placements_to_exclude)
            self.solve_DLX()
            total_solutions.extend(tuple(self.solutions_text))
        self.solutions_text = total_solutions
        return total_solutions

    def display_solution(self, solution):
        self.print_placement(self.create_solution_dict(solution))
        
    def create_solution_dict(self, solution):
        '''Override for Pentominos'''
        #Exact_Cover.display_solution(self, solution)
        # Convert the solution into a dictionary for print_placement to draw a picture
        solution_dictionary = {}
        for O_entry in solution:
            if not O_entry:
                break
            row = []
            piece = None
            first_col = O_entry
            if str(first_col.C.N) in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                piece = first_col.C.N
            else:
                row.append(first_col.C.N)
            col = first_col.R
            while col != first_col:
                if str(col.C.N) in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    piece = col.C.N
                else:
                    row.append(col.C.N)
                col = col.R
            assert piece is not None
            solution_dictionary[piece] = tuple(row)
        return solution_dictionary


    def enumerate_pentomino_orientations(self):
        '''Returns a dictionary with key=pentomino name and value=list of coordinates that represent
        all the 'orientations' of that piece.  The orientations are all centered around a specific
        block in the pentomino centered on (0,0) and it includes all rotations and flipping.

        >>> pent = Pentomino(square8by8withhole, pentominoes)
        >>> len(pent.pentomino_orientations['Z'])
        20
        >>> len(pent.enumerate_pentomino_orientations()['F'])
        40
        >>> len(pent.enumerate_pentomino_orientations()['Z'])
        20
        >>> len(pent.enumerate_pentomino_orientations()['X'])
        5
        >>> pent.enumerate_pentomino_orientations()['X']
        [((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)), ((-1, -1), (0, -2), (0, -1), (0, 0), (1, -1)), ((0, 0), (1, -1), (1, 0), (1, 1), (2, 0)), ((-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, 0)), ((-1, 1), (0, 0), (0, 1), (0, 2), (1, 1))]

        '''
        shapes = self.shapes
        # Now preprocess these initial data into a set of all possible orientations of all the shapes
        # 1. Assume each 'center' is placed at 0,0 coordinates (these coords can go negative in x and y)
        # 2. Generate a tuple with the coords of all the other '1' pieces (i.e. blocks in the shape)
        #    some of these will be negative in either x or y.
        # 3. Rotate 90 degrees around the center (0,0) for all 4 rotations and generate the tuple again
        #    each time.
        # 4. Back at the starting position, flip by mirroring around the Y axis.
        # 5. Repeat #3 for the flipped shape
        # 6. Repeat all steps above for each 'center'
        #
        # This way, in the search, we have all possible orientations already calculated and can just
        # apply them and check for overlaps or boundary overruns.
        all_orientations_dict = {} #1 entry for each shape type, each entry has the block coords for all possible orientations covering the (0,0) location
        for key in shapes:
            shape_orientations = []
            d = shapes[key]
            if d['n'] == 1:
                d['labels'] = [key,] # when there is only 1, I don't provide a labels list, so create here
            if d['limited_centers']:
                centers = d['limited_centers']
            else:
                centers = d['c']
            for center in centers:
                # translate all blocks so that 'center' is at (0,0)
                col_offset = center[0] * -1
                row_offset = center[1] * -1
                coords = tuple( ((block[0]+col_offset, block[1]+row_offset) for block in d['c']) )

                if center in d['noflip_centers']:
                    flips = [0]
                else:
                    flips = [0,1]
                for flip_flag in flips:
                    coords = self.flip_piece(coords, flip_flag)
                    if center in d['norotate_centers']:
                        rotations = [0]
                    else:
                        rotations = [0, 90, 180, 270]
                    for rotation in rotations:
                        orientation_coords = self.rotate_piece(coords, rotation)
                        #Add this orientation, but first sort the coords so that if two different
                        #center blocks create identical orientations, they will be in the same order
                        #and we can de-dupe them with the set operation at the end
                        shape_orientations.append(tuple(sorted(orientation_coords, key=operator.itemgetter(0,1))))
            de_duped = list(set(shape_orientations)) #eliminate duplicates with set
            for i in range(d['n']):
                all_orientations_dict[d['labels'][i]] =  de_duped
        return all_orientations_dict

    def print_shape(self, coords):
        '''Print the coords within a coordinate system from -5 to 5 on x and y axis'''
        print(str(coords))
        block = []
        for y in range(5, -6, -1):
            line = []
            for x in range(-5, 6, 1):
                if (x,y) == (0,0):
                    line.append('O')
                elif (x,y) in coords:
                    line.append('X')
                else:
                    line.append('.')
            block.append(''.join(line))
        print('\n'.join(block))
        print('')

    def list_all_placements(self, shape_name):
        '''List all possible ways that a shape can be fit into the given enclosure.
        The enclosure is given as a list of strings.  Each string is
        a row with either a '1' (for a valid space to put a pentomino)
        or an 'X' (for a space which must not be occupied
        
        The shape_orientations is a list of coordinates with
        all possible orientations for the shape (each 'centered'
        with one part of the shape at (0,0))

        The return value is a list of coordinate sets (in the enclosure coordinate space
        with the top-left as (0,0) and going positive to the right and down) with each
        coordinate set being the spaces occupied by the shape in a valid placement of
        the shape in the enclosure.

        >>> pent8x8 = Pentomino(square8by8withhole, pentominoes)
        >>> pent6x10 = Pentomino(rectangle10by6, pentominoes)
        >>> len(pent6x10.list_all_placements('X'))
        32
        >>> o = pent8x8.enumerate_pentomino_orientations()
        >>> count = 0
        >>> for name in o:
        ...     count += len(pent8x8.list_all_placements(name))
        ...
        >>> count
        1568

        '''
        shape_orientations = self.pentomino_orientations[shape_name]
        valid_placements = []
        ewidth = len(self.enclosure[0])
        eheight = len(self.enclosure)
        for y in range(eheight):
            for x in range(ewidth):
                for o in shape_orientations:
                    placement = self.translate_piece(o, x, y)
                    valid = True
                    for coord in placement:
                        if (coord[0] < 0) or (coord[0] >= ewidth) or (coord[1] < 0) or (coord[1] >= eheight):
                            valid = False
                        elif self.enclosure[coord[1]][coord[0]].upper() == 'X':
                            valid = False
                    if valid:
                        valid_placements.append(tuple(sorted(placement, key=operator.itemgetter(0,1))))
        valid_placements = list(set(valid_placements)) #eliminate duplicates                    
        return valid_placements

    def print_placement(self, placement_dict):
        print(self.placement_text(placement_dict))
        
    def placement_text(self, placement_dict):
        trans = {'1':'.', 'X':' '}
        board = [ [trans[self.enclosure[r][c]] for c in range(len(self.enclosure[0]))] for r in range(len(self.enclosure)) ]
        for name, coords in placement_dict.iteritems():
            for coord in coords:
                board[coord[1]][coord[0]] = name
        return '\n'.join([''.join(row) for row in board])

    def print_possible_piece_placements(self, piece):
        p = sorted(self.list_all_placements(piece), key=operator.itemgetter(0))
        for i in range(len(p)):
            print(p[i])
            self.print_placement({piece:p[i]})
            print('')

    def translate_piece(self, coords, xoffset, yoffset):
        '''Translate the given coordinates by xoffset,yoffset and return teh new coords

        >>> pent = Pentomino(square8by8withhole, pentominoes)
        >>> pent.translate_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), -2, 3)
        ((-2, 3), (-1, 3), (-1, 4), (-1, 5), (0, 5))
        '''
        return tuple( ((block[0]+xoffset, block[1]+yoffset) for block in coords) )

    def rotate_piece(self, coords, rotation):
        '''Rotate the given coordinates around the (0,0) point and return the coordinates.
        Rotation amount can be 0, 90, 180 or 270 (which are in degrees)

        >>> pent = Pentomino(square8by8withhole, pentominoes)
        >>> pent.rotate_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), 0)
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2))
        >>> pent.rotate_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), 90)
        ((0, 0), (0, -1), (1, -1), (2, -1), (2, -2))
        >>> pent.rotate_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), 180)
        ((0, 0), (-1, 0), (-1, -1), (-1, -2), (-2, -2))
        >>> pent.rotate_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), 270)
        ((0, 0), (0, 1), (-1, 1), (-2, 1), (-2, 2))

        '''
        if rotation == 0:
            return coords
        elif rotation == 90:
            return tuple( ((block[1], -1*block[0]) for block in coords) )
        elif rotation == 180:
            return tuple( ((-1*block[0], -1*block[1]) for block in coords) )
        elif rotation == 270:
            return tuple( ((-1*block[1], block[0]) for block in coords) )
        else:
            raise ValueError, "invalid rotation specified"

    def flip_piece(self, coords, flip_flag):
        '''Flip  the given coordinates by mirroring around the Y axis and return the coordinates.
        flip_flag can be 0, or 1: 0 = no flip, 1=flip
        
        >>> pent = Pentomino(square8by8withhole, pentominoes)
        >>> pent.flip_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), 1)
        ((0, 0), (-1, 0), (-1, 1), (-1, 2), (-2, 2))
        >>> pent.flip_piece(((0,0),(1,0),(1,1),(1,2),(2,2)), 0)
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2))
        '''
        if flip_flag == 0:
            return coords
        else:
            return tuple( ((-1*block[0], block[1]) for block in coords) )

    def flip_enclosure_text(self, enclosure_text):
        '''Given enclosure defined as a text string with embedded newlines,
        return a version of that string flipped over

        >>> pent3x20 = Pentomino(rectangle3x20, pentominoes)
        >>> print(pent3x20.flip_enclosure_text('\\n'.join(['abcde','fghij'])))
        edcba
        jihgf
        '''
        return '\n'.join( [ s[::-1] for s in enclosure_text.split('\n')] )

    def rotate_enclosure_text(self, enclosure_text): 
        '''Given enclosure defined as a text string with embedded newlines,
        return a version of that string rotated 90 degrees.
        Note that rotating by 90 degrees is only needed for square enclosures
        (otherwise, you can get by with just 180 degree rotation).  However,
        we will use 90 degree twice to cover all possibilities.

        >>> pent3x20 = Pentomino(rectangle3x20, pentominoes)
        >>> print(pent3x20.rotate_enclosure_text('\\n'.join(['abcde','fghij'])))
        fa
        gb
        hc
        id
        je
        '''
        strings = enclosure_text.split('\n')
        rotated = [''.join([strings[row][col]  for row in range(len(strings)-1,-1,-1)])  for col in range(len(strings[0]))]
        return '\n'.join(rotated)

    def rotate_enclosure_text_180(self, enclosure_text):
        '''Given enclosure defined as a text string with embedded newlines,
        return a version of that string rotated 180 degrees

        >>> pent3x20 = Pentomino(rectangle3x20, pentominoes)
        >>> print(pent3x20.rotate_enclosure_text_180('\\n'.join(['abcde','fghij'])))
        jihgf
        edcba
        '''
        return enclosure_text[::-1]

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

    if False: #Piece Placements
        print('Pentomino')
        pent8x8 = Pentomino(square8by8withhole, pentominoes)
        pent6x10 = Pentomino(rectangle10by6, pentominoes)

        pent6x10.print_possible_piece_placements('I')
        #pent8x8.print_possible_piece_placements('X')

        print(len(pent8x8.input_array))

    if False: # Exact_Cover object and Solve_DLX

        print('Exact Cover and solve_DLX')
        if True:
            ec = Exact_Cover(danny_ec, verbose=True, debug=global_debug)
        else:
            print('Knuth EC')
            ec = Exact_Cover(knuth_ec, verbose=True, debug=global_debug)


        t = timedcall(ec.solve_DLX)
        print('Found {} solutions in {} seconds'.format(len(ec.solutions), t))

    if False: # Pentomino 6x10
        pent6x10 = Pentomino(rectangle10by6, pentominoes, verbose=False, debug=global_debug, stopatfirstsolution=False)
        t = timedcall(pent6x10.solve_DLX)
        print('Found {} solutions in {} seconds'.format(len(pent6x10.solutions_text), t))
        outfilename = 'pentomino6x10_solutions.txt'
        f = open(outfilename, 'w')
        f.write('#'.join(pent6x10.solutions_text))
        f.close()
        print('saved solutions to {}'.format(outfilename))
        column_print(pent6x10.solutions_text, columns=6)

    if False:
        with open('pentomino6x10_solutions.txt', 'r') as f:
            lines = f.read().split('#')
        column_print(lines, fixed_width=11, columns=6)

    if False:  # 3x20
        pent3x20 = Pentomino(rectangle3x20, pentominoes, verbose=False, debug=global_debug, stopatfirstsolution=False)
        t = timedcall(pent3x20.solve_DLX)
        print('Found {} solutions in {} seconds\nHere they are:'.format(len(pent3x20.solutions_text), t))
        outfilename = 'pentomino3x20_solutions.txt'
        f = open(outfilename, 'w')
        f.write('#'.join(pent3x20.solutions_text))
        f.close()
        print('saved solutions to {}'.format(outfilename))
        column_print(pent3x20.solutions_text, fixed_width=21, columns=2)

    if False:
        pent8x8hole = Pentomino(square8by8withhole, pentominoes)
        t = timedcall(pent8x8hole.solve_DLX)
        print('Found {} solutions in {} seconds\nHere they are:'.format(len(pent8x8hole.solutions_text), t))
        outfilename = 'pentomino8x8hole_solutions.txt'
        f = open(outfilename, 'w')
        f.write('#'.join(pent8x8hole.solutions_text))
        f.close()
        print('saved solutions to {}'.format(outfilename))
        column_print(pent8x8hole.solutions_text, fixed_width=9, columns=6)

    if False: # Main puzzle I have in wood
        pent_plus_sq_8x8 = Pentomino(square8by8, pent_plus_square)
        print('Pentominos for pent_plus_sq_8x8 about to call solve_DLX')
        t = timedcall(pent_plus_sq_8x8.solve_DLX)
        print('Found {} solutions in {} seconds\nHere they are:'.format(len(pent_plus_sq_8x8.solutions_text), t))
        outfilename = 'pent_plus_sq_8x8_solutions.txt'
        f = open(outfilename, 'w')
        f.write('#'.join(pent_plus_sq_8x8.solutions_text))
        f.close()
        print('saved solutions to {}'.format(outfilename))
        column_print(pent_plus_sq_8x8.solutions_text, fixed_width=9, columns=6)

    if False: # Exclude some 'I' Placements in 6x10
        pent6x10_noborderIs = Pentomino(rectangle10by6, pentominoes, verbose=False, debug=global_debug, stopatfirstsolution=False,
                    placements_to_exclude = {'I': [ ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)), #left side (1)
                                                    ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)), #left side (2)
                                                    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)), #top (1)
                                                    ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)), #top (2)
                                                    ((2, 0), (3, 0), (4, 0), (5, 0), (6, 0)), #top (3)
                                                    ((3, 0), (4, 0), (5, 0), (6, 0), (7, 0)), #top (4)
                                                    ((4, 0), (5, 0), (6, 0), (7, 0), (8, 0)), #top (6)
                                                    ((5, 0), (6, 0), (7, 0), (8, 0), (9, 0)), #top (7)
                                                    ((9, 0), (9, 1), (9, 2), (9, 3), (9, 4)), #right side (1)
                                                    ((9, 1), (9, 2), (9, 3), (9, 4), (9, 5)), #right side (2)
                                                    ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5)), #bottom (1)
                                                    ((1, 5), (2, 5), (3, 5), (4, 5), (5, 5)), #bottom (2)
                                                    ((2, 5), (3, 5), (4, 5), (5, 5), (6, 5)), #bottom (3)
                                                    ((3, 5), (4, 5), (5, 5), (6, 5), (7, 5)), #bottom (4)
                                                    ((4, 5), (5, 5), (6, 5), (7, 5), (8, 5)), #bottom (6)
                                                    ((5, 5), (6, 5), (7, 5), (8, 5), (9, 5)), #bottom (7)
                                                   ]
                                            }
        )
        t = timedcall(pent6x10_noborderIs.solve_DLX)
        print('Found {} solutions in {} seconds'.format(len(pent6x10_noborderIs.solutions_text), t))
        outfilename = 'pentomino6x10_noborderIs_solutions.txt'
        f = open(outfilename, 'w')
        f.write('#'.join(pent6x10_noborderIs.solutions_text))
        f.close()
        print('saved solutions to {}'.format(outfilename))
        column_print(pent6x10_noborderIs.solutions_text, fixed_width=11, columns=6)
        
    if False:
        with open('pentomino6x10_noborderIs_solutions.txt', 'r') as f:
            lines = f.read().split('#')
        column_print(lines, fixed_width=11, columns=6)
        
    if False:  # Solve with one of the pieces made into a bigshape enclosure  
        #every shape can be made into an enclosure (each block becomes a 3x3 block array)
        #and solved with 9 of the other pieces.
        bigshape = 'W'
        enclosure = make_bigshape_enclosure(bigshape)
        pentominos_less_bigshape = {k:v for k,v in pentominoes.iteritems() if k != bigshape}
        pent_bigshape = Pentomino(enclosure, pentominos_less_bigshape, skip_sizecheck=True)
        t, _ = timedcall(pent_bigshape.solve_with_subset, 9)
        print('Found {} solutions in {} seconds'.format(len(pent_bigshape.solutions_text), t))
        outfilename = 'pent_{}_solutions.txt'.format(bigshape)
        f = open(outfilename, 'w')
        f.write('#'.join(pent_bigshape.solutions_text))
        f.close()
        print('saved solutions to {}'.format(outfilename))
        column_print(pent_bigshape.solutions_text, fixed_width=10, columns=6)
        
    if False:
        print('About to start Y8test')
        Y8test = Pentomino(eightYs, Y8, debug=False, verbose=True, stopatfirstsolution=True)
        print('Y11test object creation complete.')
        t = timedcall(Y8test.solve_DLX)
        print('Found {} solutions in {} seconds\nHere they are:'.format(len(Y8test.solutions_text), t))
        outfilename = 'eightYs_solutions.txt'
        f = open(outfilename, 'w')
        f.write('#'.join(Y8test.solutions_text))
        f.close()
        column_print(Y8test.solutions_text, fixed_width=9, columns=5)
        
    if False:
        print('About to start Y41test')
        Y41test = Pentomino(square15x15_predef, Y41, debug=True, verbose=True, stopatfirstsolution=True)
        print('Y41test object creation complete.  Details:\nEnclosure\n{0}\nArray\n{1}'.format(Y41test.enclosure_text, Y41test.input_array))
        if True:
            t = timedcall(Y41test.solve_DLX)
            print('Found {} solutions in {} seconds\nHere they are:'.format(len(Y41test.solutions_text), t))
            outfilename = 'Y45_in15x15_predef_solutions.txt'
            f = open(outfilename, 'w')
            f.write('#'.join(Y41test.solutions_text))
            f.close()
            column_print(Y41test.solutions_text, fixed_width=16, columns=5)
        
