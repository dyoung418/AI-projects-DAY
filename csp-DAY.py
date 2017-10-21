"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)."""

from __future__ import print_function
import copy, random, operator

#############################################################################
###      UTILITIES
#############################################################################


def count_if(predicate, seq):
    """Count the number of elements of seq for which the predicate is true.
    >>> count_if(callable, [42, None, max, min])
    2
    """
    f = lambda count, x: count + (not not predicate(x))
    return reduce(f, seq, 0)

def find_if(predicate, seq):
    """If there is an element of seq that satisfies predicate; return it.
    >>> find_if(callable, [3, min, max])
    <built-in function min>
    >>> find_if(callable, [1, 2, 3])
    """
    for x in seq:
        if predicate(x): return x
    return None

def every(predicate, seq):
    """True if every element of seq satisfies predicate.
    >>> every(callable, [min, max])
    1
    >>> every(callable, [min, 3])
    0
    """
    for x in seq:
        if not predicate(x): return False
    return True

def update(x, **entries):
    """Update a dict; or an object with slots; according to entries.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)   
    else:
        x.__dict__.update(entries) 
    return x

def if_(test, result, alternative):
    """Like C++ and Java's (test ? result : alternative), except
    both result and alternative are always evaluated. However, if
    either evaluates to a function, it is applied to the empty arglist,
    so you can delay execution by putting it in a lambda.
    >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
    'ok'
    """
    if test:
        if callable(result): return result()
        return result
    else:
        if callable(alternative): return alternative()
        return alternative

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x))

class DefaultDict(dict):
    """Dictionary with a default value for unknown keys."""
    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, copy.deepcopy(self.default))
    
    def __copy__(self):
        copy = DefaultDict(self.default)
        copy.update(self)
        return copy
    
class Struct:
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(args)

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(lt): Queue where items are sorted by lt, (default <).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self): 
        abstract

    def extend(self, items):
        for item in items: self.append(item)

def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []

class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []; self.start = 0
    def append(self, item):
        self.A.append(item)
    def __len__(self):
        return len(self.A) - self.start
    def extend(self, items):
        self.A.extend(items)     
    def pop(self):        
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x)."""
    def __init__(self, order=min, f=lambda x: x):
        update(self, A=[], order=order, f=f)
    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))
    def __len__(self):
        return len(self.A)
    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]


class Problem(object):
    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial; self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        abstract

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        abstract

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        abstract

#############################################################################
###      CSP CODE
#############################################################################
        

class CSP(Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        vars        A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases. (For example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(N^4) for the
    explicit representation.) In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP.  Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation

    In the search module, you could do: search.depth_first_graph_search(australia)
    and get:
    <Node (('WA', 'B'), ('Q', 'B'), ('T', 'B'), ('V', 'B'), ('SA', 'G'), ('NT', 'R'), ('NSW', 'R'))>
    """

    def __init__(self, vars, domains, neighbors, constraints):
        "Construct a CSP problem. If vars is empty, it becomes domains.keys()."
        vars = vars or domains.keys()
        self.vars = vars
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.initial = ()
        self.curr_domains = None
        self.nassigns = 0
##        update(self, vars=vars, domains=domains,
##               neighbors=neighbors, constraints=constraints,
##               initial=(), curr_domains=None, nassigns=0)

    def assign(self, var, val, assignment):
        "Add {var: val} to assignment; Discard the old value if any."
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        "Return the number of conflicts var=val has with other variables."
        # Subclasses may implement this more efficiently
        def conflict(var2):
            return (var2 in assignment
                    and not self.constraints(var, val, var2, assignment[var2]))
        return count_if(conflict, self.neighbors[var])

    def display(self, assignment):
        "Show a human-readable representation of the CSP."
        # Subclasses can print in a prettier way, or display with a GUI
        print('CSP: {} with assignment: {}'.format(self, assignment))

    ## These methods are for the tree- and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: nonconflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.vars):
            return []
        else:
            assignment = dict(state)
            var = find_if(lambda v: v not in assignment, self.vars)
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, (var, val)):
        "Perform an action and return the new state."
        return state + ((var, val),)

    def goal_test(self, state):
        "The goal is to assign all vars, with all constraints satisfied."
        assignment = dict(state)
        return (len(assignment) == len(self.vars) and
                every(lambda var: self.nconflicts(var, assignment[var],
                                                  assignment) == 0,
                      self.vars))

    ## These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = dict((v, list(self.domains[v]))
                                     for v in self.vars)

    def suppose(self, var, value):
        "Start accumulating inferences from assuming var=value."
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        "Rule out var=value."
        self.curr_domains[var].remove(value)
        if removals is not None: removals.append((var, value))

    def choices(self, var):
        "Return all values for var that aren't currently ruled out."
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        "Return the partial assignment implied by the current inferences."
        self.support_pruning()
        return dict((v, self.curr_domains[v][0])
                    for v in self.vars if 1 == len(self.curr_domains[v]))

    def restore(self, removals):
        "Undo a supposition and all inferences from it."
        for B, b in removals:
            self.curr_domains[B].append(b)

    ## This is for min_conflicts search

    def conflicted_vars(self, current):
        "Return a list of variables in current assignment that are in conflict"
        return [var for var in self.vars
                if self.nconflicts(var, current[var], current) > 0]

#______________________________________________________________________________
# Constraint Propagation with AC-3

def AC3(csp, queue=None, removals=None):
    """[Fig. 6.3]"""
    if queue is None:
        queue = [(Xi, Xk) for Xi in csp.vars for Xk in csp.neighbors[Xi]]
    csp.support_pruning()
    while queue:
        (Xi, Xj) = queue.pop()
        if revise(csp, Xi, Xj, removals):
            if not csp.curr_domains[Xi]:
                return False
            for Xk in csp.neighbors[Xi]:
                if Xk != Xi:
                    queue.append((Xk, Xi))
    return True

def revise(csp, Xi, Xj, removals):
    "Return true if we remove a value."
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        if every(lambda y: not csp.constraints(Xi, x, Xj, y),
                 csp.curr_domains[Xj]):
            csp.prune(Xi, x, removals)
            revised = True
    return revised

#______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering

def first_unassigned_variable(assignment, csp):
    "The default variable order."
    return find_if(lambda var: var not in assignment, csp.vars)

def mrv(assignment, csp):
    "Minimum-remaining-values heuristic."
    return argmin_random_tie(
        [v for v in csp.vars if v not in assignment],
        lambda var: num_legal_values(csp, var, assignment))

def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count_if(lambda val: csp.nconflicts(var, val, assignment) == 0,
                        csp.domains[var])

# Value ordering

def unordered_domain_values(var, assignment, csp):
    "The default value order."
    return csp.choices(var)

def lcv(var, assignment, csp):
    "Least-constraining-values heuristic."
    return sorted(csp.choices(var),
                  key=lambda val: csp.nconflicts(var, val, assignment))

# Inference

def no_inference(csp, var, value, assignment, removals):
    return True

def forward_checking(csp, var, value, assignment, removals):
    "Prune neighbor values inconsistent with var=value."
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return False
    return True

def mac(csp, var, value, assignment, removals):
    "Maintain arc consistency."
    return AC3(csp, [(X, var) for X in csp.neighbors[var]], removals)

# The search, proper

def backtracking_search(csp,
                        select_unassigned_variable = first_unassigned_variable,
                        order_domain_values = unordered_domain_values,
                        inference = no_inference):
    """[Fig. 6.5]
    >>> backtracking_search(australia) is not None
    True
    >>> backtracking_search(australia, select_unassigned_variable=mrv) is not None
    True
    >>> backtracking_search(australia, order_domain_values=lcv) is not None
    True
    >>> backtracking_search(australia, select_unassigned_variable=mrv, order_domain_values=lcv) is not None
    True
    >>> backtracking_search(australia, inference=forward_checking) is not None
    True
    >>> backtracking_search(australia, inference=mac) is not None
    True
    >>> backtracking_search(usa, select_unassigned_variable=mrv, order_domain_values=lcv, inference=mac) is not None
    True
    """

    def backtrack(assignment):
        if len(assignment) == len(csp.vars):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result

#______________________________________________________________________________
# Min-conflicts hillclimbing search for CSPs

def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic hillclimbing on the number of conflicts."""
    # Generate a complete assignment for all vars (probably with conflicts)
    csp.current = current = {}
    for var in csp.vars:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None

def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var],
                             lambda val: csp.nconflicts(var, val, current))

#______________________________________________________________________________

def tree_csp_solver(csp):
    "[Fig. 6.11]"
    n = len(csp.vars)
    assignment = {}
    root = csp.vars[0]
    X, parent = topological_sort(csp.vars, root)
    for Xj in reversed(X):
        if not make_arc_consistent(parent[Xj], Xj, csp):
            return None
    for Xi in X:
        if not csp.curr_domains[Xi]:
            return None
        assignment[Xi] = csp.curr_domains[Xi][0]
    return assignment

def topological_sort(xs, x):
    unimplemented()

def make_arc_consistent(Xj, Xk, csp): 
    unimplemented()


#############################################################################
###      EXAMPLE PROBLEMS
#############################################################################



#______________________________________________________________________________
# Map-Coloring Problems

class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all vars have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """
    def __init__(self, value): self.value = value
    def __getitem__(self, key): return self.value
    def __repr__(self): return '{Any: %r}' % self.value

def different_values_constraint(A, a, B, b):
    "A constraint saying two neighboring variables must differ in value."
    return a != b

def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions.  Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries.  This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    return CSP(neighbors.keys(), UniversalDict(colors), neighbors,
               different_values_constraint)

def parse_neighbors(neighbors, vars=[]):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors.  The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name.  If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z')
    {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    """
    dict = DefaultDict([])
    for var in vars:
        dict[var] = []
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        dict.setdefault(A, [])
        for B in Aneighbors.split():
            dict[A].append(B)
            dict[B].append(A)
    return dict

australia = MapColoringCSP(list('RGB'),
                           'SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ')

usa = MapColoringCSP(list('RGBY'),
        """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
        UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX;
        ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
        TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
        LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
        MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
        PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
        NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
        HI: ; AK: """)

france = MapColoringCSP(list('RGBY'),
        """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
        AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
        CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
        MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
        PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
        AU BO FC PA LR""")

#______________________________________________________________________________
# n-Queens Problem

def queen_constraint(A, a, B, b):
    """Constraint is satisfied (true) if A, B are really the same variable,
    or if they are not in the same row, down diagonal, or up diagonal."""
    return A == B or (a != b and A + a != B + b and A - a != B - b)

class NQueensCSP(CSP):
    """Make a CSP for the nQueens problem for search with min_conflicts.
    Suitable for large n, it uses only data structures of size O(n).
    Think of placing queens one per column, from left to right.
    That means position (x, y) represents (var, val) in the CSP.
    The main structures are three arrays to count queens that could conflict:
        rows[i]      Number of queens in the ith row (i.e val == i)
        downs[i]     Number of queens in the \ diagonal
                     such that their (x, y) coordinates sum to i
        ups[i]       Number of queens in the / diagonal
                     such that their (x, y) coordinates have x-y+n-1 = i
    We increment/decrement these counts each time a queen is placed/moved from
    a row/diagonal. So moving is O(1), as is nconflicts.  But choosing
    a variable, and a best value for the variable, are each O(n).
    If you want, you can keep track of conflicted vars, then variable
    selection will also be O(1).
    """
##    >>> len(backtracking_search(NQueensCSP(8)))
##    8
##    """
    def __init__(self, n):
        """Initialize data structures for n Queens."""
        CSP.__init__(self, range(n), UniversalDict(range(n)),
                     UniversalDict(range(n)), queen_constraint)
        self.rows = [0]*n
        self.ups = [0]*(2*n - 1)
        self.downs = [0]*(2*n - 1)
        #update(self, rows=[0]*n, ups=[0]*(2*n - 1), downs=[0]*(2*n - 1))

    def nconflicts(self, var, val, assignment):
        """The number of conflicts, as recorded with each assignment.
        Count conflicts in row and in up, down diagonals. If there
        is a queen there, it can't conflict with itself, so subtract 3."""
        n = len(self.vars)
        c = self.rows[val] + self.downs[var+val] + self.ups[var-val+n-1]
        if assignment.get(var, None) == val:
            c -= 3
        return c

    def assign(self, var, val, assignment):
        "Assign var, and keep track of conflicts."
        oldval = assignment.get(var, None)
        if val != oldval:
            if oldval is not None: # Remove old val if there was one
                self.record_conflict(assignment, var, oldval, -1)
            self.record_conflict(assignment, var, val, +1)
            CSP.assign(self, var, val, assignment)

    def unassign(self, var, assignment):
        "Remove var from assignment (if it is there) and track conflicts."
        if var in assignment:
            self.record_conflict(assignment, var, assignment[var], -1)
        CSP.unassign(self, var, assignment)

    def record_conflict(self, assignment, var, val, delta):
        "Record conflicts caused by addition or deletion of a Queen."
        n = len(self.vars)
        self.rows[val] += delta
        self.downs[var + val] += delta
        self.ups[var - val + n - 1] += delta

    def display(self, assignment):
        "Print the queens and the nconflicts values (for debugging)."
        n = len(self.vars)
        for val in range(n):
            for var in range(n):
                if assignment.get(var,'') == val: ch = 'Q'
                elif (var+val) % 2 == 0: ch = '.'
                else: ch = '-'
                print(ch, sep='', end='')
            print('    ', end='')
            for var in range(n):
                if assignment.get(var,'') == val: ch = '*'
                else: ch = ' '
                print(str(self.nconflicts(var, val, assignment))+ch, end='')
            print('')

#______________________________________________________________________________
# Sudoku

import itertools, re

def flatten(seqs): return sum(seqs, [])

easy1   = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
harder1 = '4173698.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'

class Sudoku(CSP):
    """A Sudoku problem.
    The box grid is a 3x3 array of boxes, each a 3x3 array of cells.
    Each cell holds a digit in 1..9. In each box, all digits are
    different; the same for each row and column as a 9x9 grid.
    """
    
##    >>> e = Sudoku(easy1)
##    
##    >>> e.display(e.infer_assignment())
##    . . 3 | . 2 . | 6 . .
##    9 . . | 3 . 5 | . . 1
##    . . 1 | 8 . 6 | 4 . .
##    ------+-------+------
##    . . 8 | 1 . 2 | 9 . .
##    7 . . | . . . | . . 8
##    . . 6 | 7 . 8 | 2 . .
##    ------+-------+------
##    . . 2 | 6 . 9 | 5 . .
##    8 . . | 2 . 3 | . . 9
##    . . 5 | . 1 . | 3 . .
##    
##    >>> AC3(e); e.display(e.infer_assignment())
##    True
##    4 8 3 | 9 2 1 | 6 5 7
##    9 6 7 | 3 4 5 | 8 2 1
##    2 5 1 | 8 7 6 | 4 9 3
##    ------+-------+------
##    5 4 8 | 1 3 2 | 9 7 6
##    7 2 9 | 5 6 4 | 1 3 8
##    1 3 6 | 7 9 8 | 2 4 5
##    ------+-------+------
##    3 7 2 | 6 8 9 | 5 1 4
##    8 1 4 | 2 5 3 | 7 6 9
##    6 9 5 | 4 1 7 | 3 8 2
##    
##    >>> h = Sudoku(harder1)
##    
##    >>> None != backtracking_search(h, select_unassigned_variable=mrv, inference=forward_checking)
##    True
##    """

    R3 = range(3)
    Cell = itertools.count().next
    bgrid = [[[[Cell() for x in R3] for y in R3] for bx in R3] for by in R3]
    boxes = flatten([map(flatten, brow)       for brow in bgrid])
    rows  = flatten([map(flatten, zip(*brow)) for brow in bgrid])
    cols  = zip(*rows)

    neighbors = dict([(v, set()) for v in flatten(rows)])
    for unit in map(set, boxes + rows + cols):
        for v in unit:
            neighbors[v].update(unit - set([v]))

    def __init__(self, grid):
        """Build a Sudoku problem from a string representing the grid:
        the digits 1-9 denote a filled cell, '.' or '0' an empty one;
        other characters are ignored."""
        squares = iter(re.findall(r'\d|\.', grid))
        domains = dict((var, if_(ch in '123456789', [ch], '123456789'))
                       for var, ch in zip(flatten(self.rows), squares))
        for _ in squares:
            raise ValueError("Not a Sudoku grid", grid) # Too many squares
        CSP.__init__(self, None, domains, self.neighbors,
                     self.different_values_constraint)

    def display(self, assignment):
        def show_box(box): return [' '.join(map(show_cell, row)) for row in box]
        def show_cell(cell): return str(assignment.get(cell, '.'))
        def abut(lines1, lines2): return map(' | '.join, zip(lines1, lines2))
        print('\n------+-------+------\n'.join(
            '\n'.join(reduce(abut, map(show_box, brow))) for brow in self.bgrid))

    def different_values_constraint(self, A, a, B, b):
        "A constraint saying two neighboring variables must differ in value."
        return a != b

#______________________________________________________________________________
# The Zebra Puzzle

def Zebra():
    "Return an instance of the Zebra Puzzle."
    Colors = 'Red Yellow Blue Green Ivory'.split()
    Pets = 'Dog Fox Snails Horse Zebra'.split()
    Drinks = 'OJ Tea Coffee Milk Water'.split()
    Countries = 'Englishman Spaniard Norwegian Ukranian Japanese'.split()
    Smokes = 'Kools Chesterfields Winston LuckyStrike Parliaments'.split()
    vars = Colors + Pets + Drinks + Countries + Smokes
    domains = {}
    for var in vars:
        domains[var] = range(1, 6)
    domains['Norwegian'] = [1]
    domains['Milk'] = [3]
    neighbors = parse_neighbors("""Englishman: Red;
                Spaniard: Dog; Kools: Yellow; Chesterfields: Fox;
                Norwegian: Blue; Winston: Snails; LuckyStrike: OJ;
                Ukranian: Tea; Japanese: Parliaments; Kools: Horse;
                Coffee: Green; Green: Ivory""", vars)
    for type in [Colors, Pets, Drinks, Countries, Smokes]:
        for A in type:
            for B in type:
                if A != B:
                    if B not in neighbors[A]: neighbors[A].append(B)
                    if A not in neighbors[B]: neighbors[B].append(A)
    def zebra_constraint(A, a, B, b, recurse=0):
        same = (a == b)
        next_to = abs(a - b) == 1
        if A == 'Englishman' and B == 'Red': return same
        if A == 'Spaniard' and B == 'Dog': return same
        if A == 'Chesterfields' and B == 'Fox': return next_to
        if A == 'Norwegian' and B == 'Blue': return next_to
        if A == 'Kools' and B == 'Yellow': return same
        if A == 'Winston' and B == 'Snails': return same
        if A == 'LuckyStrike' and B == 'OJ': return same
        if A == 'Ukranian' and B == 'Tea': return same
        if A == 'Japanese' and B == 'Parliaments': return same
        if A == 'Kools' and B == 'Horse': return next_to
        if A == 'Coffee' and B == 'Green': return same
        if A == 'Green' and B == 'Ivory': return (a - 1) == b
        if recurse == 0: return zebra_constraint(B, b, A, a, 1)
        if ((A in Colors and B in Colors) or
            (A in Pets and B in Pets) or
            (A in Drinks and B in Drinks) or
            (A in Countries and B in Countries) or
            (A in Smokes and B in Smokes)): return not same
        raise 'error'
    return CSP(vars, domains, neighbors, zebra_constraint)

def solve_zebra(algorithm=min_conflicts, **args):
    z = Zebra()
    ans = algorithm(z, **args)
    for h in range(1, 6):
        print('House', h, end='')
        for (var, val) in ans.items():
            if val == h: print( var, end='')
        print('')
    return ans['Zebra'], ans['Water'], z.nassigns, ans



#############################################################################
###      PENTOMINOES PROBLEM
#############################################################################

def enumerate_pentomino_orientations():
    '''Returns a dictionary with key=pentomino name and value=list of coordinates that represent
    all the 'orientations' of that piece.  The orientations are all centered around a specific
    block in the pentomino centered on (0,0) and it includes all rotations and flipping.

    >>> len(enumerate_pentomino_orientations()['F'])
    40
    >>> len(enumerate_pentomino_orientations()['Z'])
    20
    >>> len(enumerate_pentomino_orientations()['X'])
    5
    >>> enumerate_pentomino_orientations()['X']
    [((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)), ((-1, -1), (0, -2), (0, -1), (0, 0), (1, -1)), ((0, 0), (1, -1), (1, 0), (1, 1), (2, 0)), ((-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, 0)), ((-1, 1), (0, 0), (0, 1), (0, 2), (1, 1))]

    '''
    shapes = {
        'N':{'s':((1,0,0,0,0), # helps me visually see the shape, but not otherwise used
                  (1,1,0,0,0),
                  (0,1,0,0,0),
                  (0,1,0,0,0),
                  (0,0,0,0,0)),
             'c':((0,0),(0,1),(1,1),(1,2),(1,3)), #coordinates of 1s in this rotation/flip
             'limited_centers':[],  #if tuple of tuples provided, only check those coords as centers, otherwise use all as possible centers
             'noflip_centers':[],   #if tuple of tuples provided, those centers don't need to be checked in flipped state
             'norotate_centers':[], #if tuple of tuples provided, those centers don't need to be checked in rotated state
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
             },
        }
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
                coords = flip(coords, flip_flag)
                if center in d['norotate_centers']:
                    rotations = [0]
                else:
                    rotations = [0, 90, 180, 270]
                for rotation in rotations:
                    orientation_coords = rotate(coords, rotation)
                    #Add this orientation, but first sort the coords so that if two different
                    #center blocks create identical orientations, they will be in the same order
                    #and we can de-dupe them with the set operation at the end
                    shape_orientations.append(tuple(sorted(orientation_coords, key=operator.itemgetter(0,1))))
        all_orientations_dict[key] = list(set(shape_orientations)) #eliminate duplicates with set
    return all_orientations_dict

def translate(coords, xoffset, yoffset):
    '''Translate the given coordinates by xoffset,yoffset and return teh new coords

    >>> translate(((0,0),(1,0),(1,1),(1,2),(2,2)), -2, 3)
    ((-2, 3), (-1, 3), (-1, 4), (-1, 5), (0, 5))
    '''
    return tuple( ((block[0]+xoffset, block[1]+yoffset) for block in coords) )

def rotate(coords, rotation):
    '''Rotate the given coordinates around the (0,0) point and return the coordinates.
    Rotation amount can be 0, 90, 180 or 270 (which are in degrees)

    >>> rotate(((0,0),(1,0),(1,1),(1,2),(2,2)), 0)
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2))
    >>> rotate(((0,0),(1,0),(1,1),(1,2),(2,2)), 90)
    ((0, 0), (0, -1), (1, -1), (2, -1), (2, -2))
    >>> rotate(((0,0),(1,0),(1,1),(1,2),(2,2)), 180)
    ((0, 0), (-1, 0), (-1, -1), (-1, -2), (-2, -2))
    >>> rotate(((0,0),(1,0),(1,1),(1,2),(2,2)), 270)
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

def flip(coords, flip_flag):
    '''Flip  the given coordinates by mirroring around the Y axis and return the coordinates.
    flip_flag can be 0, or 1: 0 = no flip, 1=flip
    
    >>> flip(((0,0),(1,0),(1,1),(1,2),(2,2)), 1)
    ((0, 0), (-1, 0), (-1, 1), (-1, 2), (-2, 2))
    >>> flip(((0,0),(1,0),(1,1),(1,2),(2,2)), 0)
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2))
    '''
    if flip_flag == 0:
        return coords
    else:
        return tuple( ((-1*block[0], block[1]) for block in coords) )

def print_shape(coords):
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

rectangle10by6 = ['1111111111',
                   '1111111111',
                   '1111111111',
                   '1111111111',
                   '1111111111',
                   '1111111111']

square8by8withhole = ['11111111',
                       '11111111',
                       '11111111',
                       '111XX111',
                       '111XX111',
                       '11111111',
                       '11111111',
                       '11111111']

def list_all_placements(enclosure, shape_orientations):
    '''List all possible ways that a shape can be fit
    into the given enclosure.
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

    >>> len(list_all_placements(rectangle10by6, enumerate_pentomino_orientations()['X']))
    32
    >>> o = enumerate_pentomino_orientations()
    >>> count = 0
    >>> for name in o:
    ...     count += len(list_all_placements(square8by8withhole, o[name]))
    ...
    >>> count
    1568

    '''
    valid_placements = []
    ewidth = len(enclosure[0])
    eheight = len(enclosure)
    for y in range(eheight):
        for x in range(ewidth):
            for o in shape_orientations:
                placement = translate(o, x, y)
                valid = True
                for coord in placement:
                    if (coord[0] < 0) or (coord[0] >= ewidth) or (coord[1] < 0) or (coord[1] >= eheight):
                        valid = False
                    elif enclosure[coord[1]][coord[0]].upper() == 'X':
                        valid = False
                if valid:
                    valid_placements.append(tuple(sorted(placement, key=operator.itemgetter(0,1))))
    valid_placements = list(set(valid_placements)) #eliminate duplicates                    
    return valid_placements


def print_placement(enclosure, placement_dict):
    trans = {'1':'.', 'X':' '}
    board = [ [trans[enclosure[r][c]] for c in range(len(enclosure[0]))] for r in range(len(enclosure)) ]
    #board = [ ['.' for c in range(len(enclosure[0]))] for r in range(len(enclosure)) ]
    for name, coords in placement_dict.iteritems():
        for coord in coords:
            board[coord[1]][coord[0]] = name
    print('\n'.join([''.join(row) for row in board]))


def print_possible_piece_placements(enclosure, piece):
    p = sorted(list_all_placements(enclosure, enumerate_pentomino_orientations()[piece]), key=operator.itemgetter(0))
    for i in range(len(p)):
        print(i)
        print_placement(enclosure, {piece:p[i]})
        print('')


def pentomino_constraint(A, a, B, b):
    """Constraint is satisfied (true) if A, B are really the same variable,
    or if they do not share a coordinate (i.e. have no overlap)"""
    if A==B:
        return True
    else:
        for coord in a:
            if coord in b:
                return False
        return True

class N_ominoes(CSP):
    """Make a CSP for the Pentominoes problem for search 

    There will be 12 variables, one for each pentomino
    
    The Domain of possible assignments to these variables will be the
    set of coordinates for all possible placements into the enclosure
    for that pentomino

    Neighbors will be all the other Pentomino pieces

    The Constraint will state that no Pentomino can have a coordinate
    that is the same as any other pentomino.  Since we already know that
    any assignment is a valid placement on the enclosure, we don't need
    to check that in the constraint.  Also, if there are no coord overlaps
    among the pieces, we know that this implies that every spot in the
    enclosure is covered because there are only 60 spots in any
    valid enclosure.
    """
    def __init__(self, enclosure, shape_orientations):
        """Initialize data structures for Pentamino puzzle
        Orientations is a dictionary:
        {'shape_name': ( (0,0), (-1,1), etc. coords for shape centered at (0,0)}
        puzzle_shape is list of strings.  Each list is a row and the strings
        have either "X"(no piece here) or "1"(place pieces here)
        """
        variables = list(shape_orientations.keys())
        domains = {name:list_all_placements(enclosure, shape_orientations[name]) for name in variables}
        neighbors = {name: tuple(set(variables)-set(name)) for name in variables}
        CSP.__init__(self, variables, domains, neighbors, pentomino_constraint)

        self.enclosure = enclosure
        # To track ongoing conflicts, use a dictionary called 'cells'
        # The keys will be cell coordinates and the values will be a list of
        # variables which have a coordinate assignment which includes that cell.
        self.cells = {}
        for y in range(len(self.enclosure)):
            for x in range(len(self.enclosure[0])):
                if self.enclosure[y][x] == '1':
                    self.cells[(x,y)] = []

    def nconflicts(self, var, val, assignment):
        "Return the number of conflicts var=val has with other variables."
        return count_if(lambda coord: len(self.cells[coord])>0, val)

    def assign(self, var, val, assignment):
        "Assign var, and keep track of conflicts."
        if var in assignment:
            self.unassign(var, assignment)
        assignment[var] = val
        if debug: print('Assign {}'.format(var))
        for coord in val:
            self.cells[coord].append(var)
        if debug: self.display(assignment)

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            if debug: print('Remove {}'.format(var))
            oldval = assignment[var]
            del assignment[var]
            for coord in oldval:
                if var in self.cells[coord]:
                    self.cells[coord].remove(var)

    def display(self, assignment):
        if assignment:
            trans = {'1':'.', 'X':' '}
            board = [ [trans[self.enclosure[r][c]] for c in range(len(self.enclosure[0]))] for r in range(len(self.enclosure)) ]
            for name, coords in assignment.iteritems():
                for coord in coords:
                    board[coord[1]][coord[0]] = name
            print('\n'.join([''.join(row) for row in board]))
            print('')
        else:
            print('assignment is None')



#############################################################################
###      RUN
#############################################################################


if __name__ == '__main__':

    if True:
        import doctest
        debug = False
        print('\nRunning Doctest...')
        doctest.testmod(verbose=False)
        print('\n\n')


    show_examples = False
    if show_examples:
        print('\n*****    Australia - min_conflicts ')   
        assignment = min_conflicts(australia)
        australia.display(assignment)

        print('\n*****    Australia - backtracking_search (select_unassigned_variable=mrv) ')   
        australia.display(backtracking_search(australia, select_unassigned_variable=mrv))
        print('\n*****    Australia - backtracking_search (order_domain_values=lcv) ')   
        australia.display(backtracking_search(australia, order_domain_values=lcv))
        print('\n*****    Australia - backtracking_search (select_unassigned_variable=mrv, order_domain_values=lcv) ')   
        australia.display(backtracking_search(australia, select_unassigned_variable=mrv, order_domain_values=lcv))
        print('\n*****    Australia - backtracking_search (inference=forward_checking) ')   
        australia.display(backtracking_search(australia, inference=forward_checking))
        print('\n*****    Australia - backtracking_search (inference=mac) ')   
        australia.display(backtracking_search(australia, inference=mac))
        print('\n*****    USA - backtracking_search (select_unassigned_variable=mrv, order_domain_values=lcv, inference=mac) ')   
        australia.display(backtracking_search(usa, select_unassigned_variable=mrv, order_domain_values=lcv, inference=mac))

        print('\n*****    NQueens Problem (8x8) - backtracking_search ')   
        print(len(backtracking_search(NQueensCSP(8))))
        queens8 = NQueensCSP(8)
        queens8.display(backtracking_search(queens8))
        #print('*****    NQueens Problem (8x8) - min_conflicts ')   
        #queens8.display(min_conflicts(queens8))

        print('\n*****    Sudoku (easy) AC3 constraint propagation ')   
        e = Sudoku(easy1)
        e.display(e.infer_assignment())
        AC3(e)
        e.display(e.infer_assignment())
        print('\n*****    Sudoku (hard) AC3 (fails - inference alone cannot solve) ')   
        h = Sudoku(harder1)
        h.display(h.infer_assignment())
        AC3(h)
        h.display(h.infer_assignment())
        print('\n*****    Sudoku (hard) backtracking_search (succeeds) ')   
        h.display(backtracking_search(h, select_unassigned_variable=mrv, inference=forward_checking))

        print('\n*****    Zebra puzzle backtracking_search')   
        solve_zebra(algorithm=backtracking_search, select_unassigned_variable=mrv, inference=forward_checking)
        print('\n*****    Zebra puzzle AC3 constraint propagation (fails -- only propogates 3 constraints)')
        z = Zebra()
        AC3(z)
        z.display(z.infer_assignment())
        #print('\n*****    Zebra puzzle min_conflicts')   
        #solve_zebra(algorithm=min_conflicts)

    print('Pentaminoes\n')
    pent = N_ominoes(square8by8withhole, enumerate_pentomino_orientations(), )
    print('\n*****    Pentamino rectangle10by6 - backtracking_search ')   
    pent.display(backtracking_search(pent))
    print('\n*****    Pentamino rectangle10by6 - backtracking_search(select_unassigned_variable=mrv, order_domain_values=lcv)')   
    pent.display(backtracking_search(pent, select_unassigned_variable=mrv, order_domain_values=lcv))

