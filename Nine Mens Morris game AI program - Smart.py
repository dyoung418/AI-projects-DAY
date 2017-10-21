'''
Game of Nine Men's Morris.

Board

 nw-------nm------ne
 |        |        |
 |  nw---nm---ne   |
 |  |     |    |   |
 |  |   +-+-+  |   |
 mw-mw--+   +--me-me
 |  |   +-+-+  |   |
 |  |     |    |   |
 |  sw---sm---se   |
 |        |        |
 sw------sm-------se

Note that the inner square is not labeled like the outer and middle squares
but has the same vertices.
The board will be represented by a list of 3 lists.  Each of the inner
lists represents one of the squares (starting with the outer and going in)
and lists the vertices in the following order:[nw, nm, ne, me, se, sm, sw, mw]
(with this ordering, neighbors on the board are the neighbors in the [wrapped]
list plus the same location in neighboring squares if the vertex is a middle
vertex).
The values in these lists will be None if unoccupied, 0 if occupied by player 1
and 1 if occupied by player 2.

"Mille"s can be formed by 3 stones in a row connected by a line.
'''
from __future__ import print_function
#import games # Custom module from AIMA
import random
import functools
import itertools
import collections
import time

###############################################################################
##  Decorators
###############################################################################
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
    argnames = f.func_code.co_varnames[:f.func_code.co_argcount]
    fname = f.__name__

    indent = '   '
    def _f(*args, **kwargs):
        signature = '%s(%s)' % (fname, ', '.join('%s=%r' % entry
            for entry in zip(argnames,args) + kwargs.items()))
        #signature = '%s(%s, %s)' % (f.__name__, ', '.join(map(repr, args)),
        #                               ', '.join( (str(key)+'='+str(arg) for key,arg in kwargs.items()) ))
        print('%s--> %s' % (trace.level*indent, signature))
        trace.level += 1
        try:
            # your code here
            result = f(*args, **kwargs)
            print('%s<-- %s == %s' % ((trace.level-1)*indent, 
                                      signature, result))
        finally:
            trace.level -= 1
            # your code here
        #return f(*args)# DAY, this used to read 'return f(*args)' but it calls f twice that way
        return result # DAY, this used to read 'return f(*args)' but it calls f twice that way
    trace.level = 0
    return _f

def disabled(f): return f
#trace = disabled


#Helper functions

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

##def argmax(seq, fn):
##    """Return an element with highest fn(seq[i]) score; tie goes to first one.
##    >>> argmax(['one', 'to', 'three'], len)
##    'three'
##    """
##    return argmin(seq, lambda x: -fn(x))


def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    Note that it is tempting to think that this is equivalent to
    max(map(fn, seq)), but that expression returns the max fn(n) for all n in seq.
    We instead want the 'n' that corresponds to the max fn(n) for all n in seq.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    best = seq[0]; best_score = fn(best)
    if len(seq) == 1:
        return best
    for x in seq[1:]: # can't you say: for x in seq[1:]  ??
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best
    
##def argmax_list(seq, fn):
##    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
##    >>> argmax_list(['one', 'three', 'seven'], len)
##    ['three', 'seven']
##    """
##    return argmin_list(seq, lambda x: -fn(x))

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'to', 'three', 'or', 'seven'], len)
    ['three', 'seven']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best

##def argmax_random_tie(seq, fn):
##    "Return an element with highest fn(seq[i]) score; break ties at random."
##    return argmin_random_tie(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    """Return an element with highest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmax_random_tie(s, f) in argmax_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best


###############################################################################
##  Generic Games Functions
###############################################################################

#Game abstract class
class Game(object):
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def __init__(self):
        '''Initialize the game variables.
        Must include the following:
        self.initial
        '''
        abstract

    def actions(self, state):
        "Return a list of the allowable moves at this point."
        abstract

    def result(self, state, move):
        "Return the state that results from making a move from a state."
        abstract

    def utility(self, state, player):
        "Return the value of this final state to player."
        abstract

    def utilities(self, state):
        "Return the value of this final state for each of the players, returned in a tuple"
        raise NotImplementedError

    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        return not self.actions(state)

    def num_players(self):
        "Return the number of players in this game"
        raise NotImplementedError

    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state.to_move

    def display(self, state):
        "Print or otherwise display the state."
        print(state)

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

# Main play_game function

def play_game(game, *players):
    '''Play an n-person game which allows for non-alternating moves (i.e. when
    a game allows a player to earn a second move, for example).
    "players" are functions that take game and state as arguments and return a move
    '''
    state = game.initial
    try:
        while True:
            player = players[game.to_move(state)] #depends on to_move() to not go out of bounds for *players
            if debug: print("*********** NEW TURN: PLAYER {} *************".format(game.to_move(state)))
            move = player(game, state)
            state = game.result(state, move)
            game.display(state) #DAY DEBUG
            if game.terminal_test(state):
                return game.utility(state, game.to_move(game.initial))
    except KeyboardInterrupt:
        print('Game Interrupted -- printing board')
        game.display(state)
        raise

def play_move_alternating_game(game, *players):
    '''Play an n-person, move-alternating game.
    "players" are functions that take game and state as arguments and return a move
    '''
    state = game.initial
    try:
        while True:
            for player in players:
                move = player(game, state)
                state = game.result(state, move)
                game.display(state) #DAY DEBUG
                if game.terminal_test(state):
                    return game.utility(state, game.to_move(game.initial))
    except KeyboardInterrupt:
        print('Game Interrupted -- printing board')
        game.display(state)
        raise

#Players
def alphabeta_player_maker(game, state, d=None, t=None, multi=False):
    '''Note that play_game will never supply d= or t= arguments, the
    player signature is (game, state).

    d=None, t=None: Does a alphabeta_full_search out to every leaf node
                    and only calls utility() on leaf nodes.
    d=4, t=None:    Will do a depth-limited search out to a depth of 4
                    (actually it will evaluate states at depth 5 since the
                    tests is depth>d). It calls game.utility() as a state
                    evaluation function at the nodes at the depth limit.
    d=None, t=10:   Will do a time-limited iterative deepening search to
                    a limit of 10 seconds (recall that iterative deepening
                    is not just calling off at depth-first search after X
                    seconds; rather, it is doing a depth-first search of
                    depth=1, then if time is not out, doing a new
                    depth-first search of depth=2, then if time is not out
                    going to d=3, etc.  So on each iteration, you get in the
                    full *breadth* of each ply down to the depth limit.  If
                    you just time-limited a depth-first search, you may
                    only have time to go very deep on the first move
                    of ply 0.)
    d=4, t=10:      Will do a time-limited iterative deepening search to
                    either a limit of 10 seconds, or a depth of 4, whichever
                    comes first.'''
    return alphabeta_iterat_deepening_search(game, state, d=d, t=t, multi=multi)

#alphabeta_player_default = functools.partial(alphabeta_player_maker, d=4, t=30)
def alphabeta_player_default(game, state):
    return alphabeta_player_maker(game, state, d=4, t=30)

def alphabeta_player_t10(game, state):
    return alphabeta_player_maker(game, state, t=10)

def alphabeta_player_d2(game, state):
    return alphabeta_player_maker(game, state, d=2)

def random_player(game, state):
    "A player that chooses a legal move at random."
    return random.choice(game.actions(state))



# Search algorithms

def minimax_decision(game, state):
    """Given a state in a two-player game, calculate the best move by searching
    forward all the way to the terminal states. [Fig. 5.3]
    At each depth of the game tree in which it is the player's turn, choose
    the move with the max expected utility where the expected utility of a node
    is the minimum expected utility for the child nodes (because the opponent
    will choose that child node such to minimize player's utility)
    """
    infinity = 1.0e400
    player = game.to_move(state)

    def max_value(state):
        '''Note that max_value returns the utility value, not the element
        that caused that utility value.  That is why we can't just call
        return(max_value(...)) at the bottom of minimax_search'''
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    if debug:
        sorted_choices = sorted([(action, min_value(game.result(state, action))) for action in game.actions(state)], key=lambda a: a[1], reverse=True)
        print(sorted_choices)
        return sorted_choices[0][0]
    else:
        return argmax(game.actions(state), lambda a: min_value(game.result(state, a)))

def alphabeta_full_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Fig. 5.7], this version searches all the way to the leaves."""

    infinity = 1.0e400
    player = game.to_move(state)

    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_full_search:
    
    if debug:
        sorted_choices = sorted([(action, min_value(game.result(state, action),-infinity,infinity)) for action in game.actions(state)], key=lambda a: a[1], reverse=True)
        print(sorted_choices)
        return sorted_choices[0][0]
    else:
        return argmax(game.actions(state),
                  lambda a: min_value(game.result(state, a), -infinity, infinity))


def alphabeta_search(game, state, t=None, d=None, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search at either a time or depth limit
    and uses an evaluation function.  Note the time limit here is not
    iterative deepening search, it simply stops the depth-first search
    when time runs out."""

    infinity = 1.0e400
    player = game.to_move(state)
    t0 = time.clock()
    def default_cutoff_test(state, depth):
        if game.terminal_test(state):
            return game.terminal_test(state)
        elif t and time.clock() - t0 > t:
            return True
        elif d and depth > d:
            return True
        else:
            return False
    cutoff_test = (cutoff_test or default_cutoff_test)
    eval_fn = eval_fn or (lambda state: game.utility(state, player))

    def max_value(state, alpha, beta, depth):
        '''Return a tuple of (utility, action) for the max valued utility and its
        associated action'''
        #When I'm here, I am the player (MAX)
        if cutoff_test(state, depth):
            return (eval_fn(state), None)
        v = (-infinity, None) #utility/action tuple
        for a in game.actions(state):
            v = max(v, (min_value(game.result(state, a), alpha, beta, depth+1)[0], a) ) # v is the highest value so far (tuples are ordered by first element)
            if v[0] >= beta: # check if we can prune
                # My v is already larger than the best alternative that MIN (my opponent) has seen so far, so
                # she is definitely not going to choose this v over her lower option (because MIN minimizes).
                # Since any subsequent node I look at will be ignored unless it is even larger than v (since I
                # maximize), it makes no sense to go on, because MIN will reject all such larger numbers.
                #if debug: print('----PRUNE!-----')
                return v # prune by returning before end of for loop. (v isn't necesarily "right" but it doesn't matter)
            alpha = max(alpha, v[0]) # we didn't prune. Update alpha with the best option (for me, MAX) so far.
        return v

    def min_value(state, alpha, beta, depth):
        #When I'm here, I am the opponent (MIN)
        #Remember that all these utility values are from the point of view of MAX, so as
        #Player MIN, I want to minimize the values that I see as my choices.
        if cutoff_test(state, depth):
            return (eval_fn(state), None)
        v = (infinity, None) #utility/action tuple
        for a in game.actions(state):
            v = min(v, (max_value(game.result(state, a), alpha, beta, depth+1)[0], a) )
            if v[0] <= alpha: #check if we can prune
                # I already have a value which is lower than the best alternative option that MAX (my opponent) as seen
                # so far.  Since I would only be lowering this v value (since I minimize), anything I come up with is
                # either going to be the current v or lower and will be rejected by MAX, who has a higher option in alpha.
                # So return the current v (even though it may not be the exact right value for the node, but it doesn't matter)
                #if debug: print('----PRUNE!-----')
                return v
            beta = min(beta, v[0]) #update beta to the best (lowest) option I (MIN) have seen so far
        return v

    # Body of alphabeta_search starts here:
    max_util_and_action = max_value(state, -infinity, infinity, 0)
    return max_util_and_action[1] #return the associated action with the max value utility

## DAY: found bug in this implementation below:  The top node ("A" in figure 5.2) doesn't go into a call to max_value, instead, the
## top node only goes to argmax.  The problem with that is that when the first value in Figure 5.2 filters
## up to the top node, alpha should be updated to be 3 (as it would be if the top node were called with
## max_value instead of argmax).  As it is, argmax just goes on to call the next child of the top node with
## alpha and beta at -infinity and infinity, respectively, so the nodes under "B" will not be pruned as they
## should be.  Those nodes under "B" (4, and 6) are only pruned if B is called in min_value with alpha=3 as
## it would be if the top node had been called with max_value instead of argmax.
## the pseudocode on page 170 doesn't make this same mistake      

##    #if debug:
##    if False: #DEBUG DAY        
##        sorted_choices = sorted([(action, min_value(game.result(state, action),-infinity,infinity, 0)) for action in game.actions(state)], key=lambda a: a[1], reverse=True)
##        print(sorted_choices)
##        return sorted_choices[0][0]
##    else:
##        return argmax(game.actions(state), lambda a: min_value(game.result(state, a), -infinity, infinity, 0))
##        return argmax_random_tie(game.actions(state),
##                  lambda a: min_value(game.result(state, a), -infinity, infinity, 0))


def multiplayer_search(game, state, t=None, d=None, cutoff_test=None, eval_fn=None):
    '''Given a state in a multi-player game, calculate the best move by
    searching forward all the way to terminal states.  At each depth
    of the game tree, choose the move which maximizes the utility for the player
    whose move is being played.
    
    In normal minmax, the scores are always calculated from the point of view of the player at the top
    of the search tree (no matter whose turn it is) and that is why it flips between max and min.  For this
    version, however, we are always calculating the utility from the perspective
    of the player at the current node (not the top node), so we always maximize.

    This version is also useful for games where the same player can earn an extra
    turn (and thus alternating between MAX and MIN is not appropriate).

    However, alphabeta pruning does not work with multiplayer utilities.  Let's say that
    in a 4 player game, one node is already occupied with [15, 2, 9, 5] (chosen by player 3 at
    utility 9).  Player 3's alpha would be set at 9 (his best option so far).  Later, player
    4 cannot prune a utility of [4, 3, 5, 9] even though player 3's utility of 5 is lower than
    their alpha of 9 because at the next node, player 4 may uncover [10,8, 15, 20] which is better
    for both player 4 and player 3.
    
    Note that for this version of minmax, we need game.utilities() which returns a
    tuple of utilities -- one for each player, where the player number indexes the
    correct utility. (as opposed to game.utility())'''
    infinity = 1.0e400
    t0 = time.clock()
    def default_cutoff_test(state, depth):
        if game.terminal_test(state):
            return game.terminal_test(state)
        elif t and time.clock() - t0 > t:
            #if debug: print('**TIMED OUT**')
            return True
        elif d and depth > d:
            #if debug: print('**DEPTHED OUT**')
            return True
        else:
            return False
    cutoff_test = (cutoff_test or default_cutoff_test)
    eval_fn = eval_fn or (lambda state: game.utilities(state))

    
    def max_val(state, depth):
        '''Return the max utility value for the player whose move it is during
        "state".'''
        player = game.to_move(state)
        if cutoff_test(state, depth):
            return [eval_fn(state), None] # [ (player utilities), action]
        else:
            v = [(-infinity,)*game.num_players(), None]
            for a in game.actions(state):
                next_node_value = [max_val(game.result(state, a), depth+1)[0], a]
                if next_node_value[0][player] > v[0][player]:
                    v = next_node_value # v is the best utility so far (for current player)
            return v

    # Body of multiplayer_search starts here:
    max_util_and_action = max_val(state, 0)
    return max_util_and_action[1] #return the associated action with the max value utility


def alphabeta_multiplayer(game, state, t=None, d=None, cutoff_test=None, eval_fn=None, alpha_pruning=False):
    '''Given a state in a multi-player game, calculate the best move by
    searching forward using alpha-beta pruning.  Supports ending at a certain
    depth=d, or after time elapsed=t.  Uses estimation of utility value for
    non-terminal nodes and terminal nodes.  If cutoff_test is supplied, it
    overrides the 't' and 'd' arguments.
    At each depth of the game tree, choose the move which maximizes the utility
    for the player whose move is being played.
    
    In normal minmax, the scores are always calculated from the point of view of the player at the top
    of the search tree (no matter whose turn it is) and that is why it flips between max and min.  For this
    version, however, we are always calculating the utility from the perspective
    of the player at the current node (not the top node), so we always maximize.

    This version is also useful for games where the same player can earn an extra
    turn (and thus alternating between MAX and MIN is not appropriate).

    However, alphabeta pruning does not work with multiplayer utilities UNLESS WE GUARANTEE
    THAT AN INCREASE IN ONE PLAYER'S UTILITY IS ALWAYS PAIRED WITH A DECREASE IN OTHER
    PLAYER'S UTILITY.  Otherwise, let's say that
    in a 4 player game, one node is already occupied with [15, 2, 9, 5] (chosen by player 3 at
    utility 9).  Player 3's alpha would be set at 9 (his best option so far).  Later, player
    4 cannot prune a utility of [4, 3, 5, 9] even though player 3's utility of 5 is lower than
    their alpha of 9 because at the next node, player 4 may uncover [10,8, 15, 20] which is better
    for both player 4 and player 3.
    
    Note that for this version of minmax, we need game.utilities() which returns a
    tuple of utilities -- one for each player, where the player number indexes the
    correct utility. (as opposed to game.utility())'''
    infinity = 1.0e400
    t0 = time.clock()

    def default_cutoff_test(state, depth):
        if game.terminal_test(state):
            return game.terminal_test(state)
        elif t and time.clock() - t0 > t:
            return True
        elif d and depth > d:
            return True
        else:
            return False
    cutoff_test = (cutoff_test or default_cutoff_test)
    eval_fn = eval_fn or (lambda state: game.utilities(state))

    @trace
    def max_val(state, alpha_values, depth):
        '''Return the max utility value for the player whose move it is during
        "state".'''
        player = game.to_move(state)
        if cutoff_test(state, depth):
            return [eval_fn(state), None] #[ (player utilities tuple), action]
        # The "v"s we will be tracking are of the following format:
        #  [ (tuple of utilities -- one for each player), action_which_results_in_these_utilities]
        v = [(-infinity,)*game.num_players(), None] 
        for a in game.actions(state):
            next_node_value = [max_val(game.result(state, a), alpha_values, depth+1)[0], a]
            if next_node_value[0][player] > v[0][player]:
                v = next_node_value # v is the best utility so far (for current player)

            # NOTE!  Pruning doesn't work in multiplayer UNLESS WE GUARANTEE THAT
            #        AN INCREASE IN ONE PLAYER'S UTILITY IS ALWAYS PAIRED WITH A
            #        DECREASE IN *ALL* OTHER PLAYER'S UTILITY.
            # Use with Caution!
            if alpha_pruning or debug:
                for p in range(game.num_players()):
                    if p == player:
                        continue
                    else:
                        if v[0][p] < alpha_values[p]:
                            # Other player already has a best-seen utility (alpha) that is
                            # greater than the utility here.  UNDER THE ASSUMPTION that any
                            # subsequent utility I may see can only be greater for me if it
                            # is less for the other players, I can safely prune here because
                            # nothing will be better for me while overcoming the best-seen
                            # value that the other player has.
                            if debug:
                                if alpha_pruning: print(' '*10 + '----PRUNE!-----')
                                else: print(' '*10 + '---- WOULD HAVE PRUNED!-----')
                            if alpha_pruning:
                                return v #prune by returning early

                #did not prune, but update alpha_values
                alpha_values[player] = max(alpha_values[player], v[0][player])
        return v

    # Body of alphabeta_search starts here:
    max_util_and_action = max_val(state, [-infinity,]*game.num_players(), 0)
    return max_util_and_action[1] #return the associated action with the max value utility

def alphabeta_iterat_deepening_search(game, state, t=None, d=None, cutoff_test=None, eval_fn=None, multi=False):
    '''Search game to determine best action; use alpha-beta pruning.
    d=None, t=None: Does a alphabeta_full_search out to every leaf node
                    and only calls eval_fn() on leaf nodes.
    d=4, t=None:    Will do a depth-limited search out to a depth of 4
                    (actually it will evaluate states at depth 5 since the
                    tests is depth>d). It calls eval_fn() as a state
                    evaluation function at the nodes at the depth limit.
    d=None, t=10:   Will do a time-limited iterative deepening search to
                    a limit of 10 seconds (recall that iterative deepening
                    is not just calling off at depth-first search after X
                    seconds; rather, it is doing a depth-first search of
                    depth=1, then if time is not out, doing a new
                    depth-first search of depth=2, then if time is not out
                    going to d=3, etc.  So on each iteration, you get in the
                    full *breadth* of each ply down to the depth limit.  If
                    you just time-limited a depth-first search, you may
                    only have time to go very deep on the first move
                    of ply 0.)
    d=4, t=10:      Will do a time-limited iterative deepening search to
                    either a limit of 10 seconds, or a depth of 4, whichever
                    comes first.'''
    
    if t and d:
        t0 = time.clock()
        for depth in range(1, d+1):
            if multi:
                result = multiplayer_search(game, state, t=t*2, d=depth, cutoff_test=cutoff_test, eval_fn=eval_fn)
            else:
                result = alphabeta_search(game, state, t=t*2, d=depth, cutoff_test=cutoff_test, eval_fn=eval_fn)
            t1 = time.clock()
            #print(t1-t0) #debug
            if t1-t0 > t:
                print('Iterative deepening with t={},d={}, GOT TO level {}'.format(t, d, depth))
                return result
        print('Iterative deepening with t={},d={}, TRIGGERED DEPTH LIMIT at t={}'.format(t, d, time.clock()-t0))
        return result
    elif t: # d=None
        t0 = time.clock()
        while True:
            if multi:
                result = multiplayer_search(game, state, t=t*2, d=None, cutoff_test=cutoff_test, eval_fn=eval_fn)
            else:
                result = alphabeta_search(game, state, t=t*2, d=None, cutoff_test=cutoff_test, eval_fn=eval_fn)
            t1 = time.clock()
            if t1-t0 > t:
                print('Iterative deepening with t={}, GOT TO level {}'.format(t, depth))
                return result
    else: # t=None AND either d=value OR d=None
        if multi:
            return multiplayer_search(game, state, t=None, d=d, cutoff_test=cutoff_test, eval_fn=eval_fn)
        else:
            return alphabeta_search(game, state, t=None, d=d, cutoff_test=cutoff_test, eval_fn=eval_fn)
            
 


###############################################################################################################
##  Game-Specific Functions
###############################################################################################################


class NineMensMorris(Game):
    '''The game of nine men's morris'''
    player1, player2 = (0, 1)
    other = {player2:player1, player1:player2}
    State = collections.namedtuple('State', ['player', 'board', 'bstones', 'unplaced', 'removeok', 'resign'])
    NONE = 3 #use this for empty board spots

    def __init__(self, initial_state=None):
        if initial_state:
            self.initial = initial_state
        else:
            self.initial = self.State(player=self.player1, board=tuple( tuple(self.NONE for _ in range(8)) for _ in range(3) ),
                              bstones=(0,0), unplaced=(9,9), removeok=False, resign=None)
        self.acts = {'Place':self.place, 'Move':self.move, 'Remove':self.remove,
                     'Resign':self.resign}
        self.loss_condition = 2 # 2 or fewer stones == lose
        self.n_players = 2 # always 2 players for NineMensMorris
        self.last_state = self.initial
        # Initiate a Neighbors list
        def _initiate_neighbors(state):
            '''neighbors is a dictionary which has keys of every location on the board and
            values as a list of tuples which are that location's neighbors.'''
            neighbors = dict()
            board = state.board
            sqs = len(board)
            vs = len(board[0])
            for s in range(sqs):
                for v in range(vs):
                    if v % 2 == 0: #corner vertex
                        neighbors[(s,v)] = [ (s,(v+1)%8), (s,(v-1)%8) ]
                    else: #middle vertex
                        if s == 0:
                            neighbors[(s,v)] = [ (s,(v+1)%8), (s,(v-1)%8), ((s+1)%sqs,v) ]
                        elif s == sqs-1:
                            neighbors[(s,v)] = [ (s,(v+1)%8), (s,(v-1)%8), ((s-1)%sqs,v) ]
                        else:
                            neighbors[(s,v)] = [ (s,(v+1)%8), (s,(v-1)%8), ((s+1)%sqs,v), ((s-1)%sqs,v) ]
            return neighbors
        self.neighbors = _initiate_neighbors(self.initial)

        def _initiate_MilleList(state):
            '''Return a list of all triplet locations which can form milles.
            This can be used for analyzing a board, for example, to count
            how many 2-in-a-row sets there are in a configuration.  You'd want
            to do such a thing only within the possible sets of milles.'''
            milles = []
            board = state.board
            sqs = len(board)
            vs = len(board[0])
            for s in range(sqs):
                for v in range(vs):
                    if v % 2 == 0: #corner vertex
                        milles.append([(s,v), (s,(v+1)%8), (s,(v+2)%8)])
                    elif s == 0: #middle vertex and the outer square
                        milles.append([( (s+i)%8, v) for i in range(sqs)])
            return milles
        self.millelist = _initiate_MilleList(self.initial)
        self.connected_milles = set()
        for m1 in self.millelist:
            for m2 in self.millelist:
                if m1 == m2: continue
                if set(m1) & set(m2): #do they have anything in their intersection?
                    self.connected_milles.add( tuple(sorted( (tuple(m1), tuple(m2)) )) ) #sort so we don't think reverse order is a different element
        #print(repr(self.connected_milles))            

    def actions(self, state):
        "Return a list of the allowable moves at this point."
        (me, board, bstones, unplaced, removeok, resign) = state
        possible_moves = []
        #####  Remove  #####
        if removeok:
            action = 'Remove'
            opponent_stones = self.player_stones(state, self.other[me])
            eligible_stones = [ loc for loc in opponent_stones if not self.creates_mille(state, self.other[me], loc) ]
            if len(eligible_stones) < 1:
                eligible_stones = opponent_stones #there is debate on whether it is OK to remove Mille stones if that is all that is available: this implementation says yes, it is.
            return [ ( action, (location,) ) for location in eligible_stones ]
        #####  Place   #####
        elif unplaced[me] > 0:
            action = 'Place'
            empty_locs = self.player_stones(state, self.NONE)
            return [ ( action, (location,) ) for location in empty_locs ]
        #####  Resign because too few stones #####
        elif bstones[me] <= self.loss_condition:
            return [('Resign', ('Lost because number of stones is less than {}'.format(self.loss_condition),) )]
        #####  Move (or Resign if unable to move)    #####
        else: 
            action = 'Move'
            move_actions = []
            if bstones[me] == self.loss_condition + 1: # special case, when only 3 stones left, may move anywhere
                for start_stone in self.player_stones(state, me):
                    for blank_spot in self.player_stones(state, self.NONE):
                        move_actions.append( (action, (start_stone, blank_spot)) )
                return move_actions
            else: # normal move -- must move to neighboring space
                for start_stone in self.player_stones(state, me):
                    for n in self.neighbors[start_stone]:
                        if board[n[0]][n[1]] is self.NONE:
                            move_actions.append( (action, (start_stone, n)) )
                if len(move_actions) < 1:
                    return [('Resign', ('Lost because player is unable to make a move',) )]
                else:
                    return move_actions

    def result(self, state, move):
        "Return the state that results from making a move from a state."
        action, arguments = move
        return self.acts[action](state, *arguments)

    #@trace
    #@memo
    def utility(self, state, player):
        "Return the value of this state to player. (Between -1 and 1)"
        (state_player, board, bstones, unplaced, removeok, resign) = state
        me = player
        opponent = self.other[me]
        #if debug: print('utility: board={}'.format(state.board), end='')
        ## Check if already resigned
        if resign:
            if resign[0] == player:
                return -1
            elif resign[0] == opponent:
                return 1
        ## Win or loss utilities because 2 or fewer stones
        if bstones[me] + unplaced[me] <= self.loss_condition:
            return -1
        elif bstones[opponent] + unplaced[opponent] <= self.loss_condition:
            return 1
        ## Win or loss utilities because unable to move
        liberties = {me:0, opponent:0}
        for loc in self.player_stones(state, me):
            liberties[me] += self.liberties(state, loc)
        for loc in self.player_stones(state, opponent):
            liberties[opponent] += self.liberties(state, loc)
        if unplaced[me] == 0 and liberties[me] == 0:
            return -1
        if unplaced[opponent] == 0 and liberties[opponent] == 0:
            return 1
        ## Non-terminal state utility

        stone_count = {me:bstones[me]+unplaced[me], opponent:bstones[opponent]+unplaced[opponent]}
        if removeok: stone_count[self.other[state_player]] -= 1
        unblocked_doubles = {me:0, opponent:0}
        milles = {me:0, opponent:0}
        potential_double_double = {me:0, opponent:0}
        alternating_milles = {me:0, opponent:0} #where one stone can move back/forth between two milles
        for row in self.millelist:
            stones_in_row = [board[loc[0]][loc[1]] for loc in row]
            my_stones = stones_in_row.count(me)
            opponent_stones = stones_in_row.count(opponent)
            if opponent_stones == 0:
                if my_stones == 2:
                    unblocked_doubles[me] += 1
                elif my_stones == 3:
                    milles[me] += 1
                    for location in row:
                        if self.can_move_to_mille(state, location):
                            alternating_milles[me] += 1
            elif my_stones == 0:
                if opponent_stones == 2:
                    unblocked_doubles[opponent] += 1
                elif opponent_stones == 3:
                    milles[opponent] += 1
                    for location in row:
                        if self.can_move_to_mille(state, location):
                            alternating_milles[opponent] += 1
        # a "2 single rows connected by a blank" category (i.e. with one stone it becomes two doubles.)


        ## DAY TODO this pot_dbl_dbl code isn't working.  It gives player 1 credit for 2 of them
        # in the following: board=((0,0,1,0,0,1,_,_),(_,_,_,_,_,_,_,_),(_,_,_,_,_,_,1,1))
        # when I can manually only see 1
        
        for cmil in self.connected_milles:
            stones_in_cmil = [board[loc[0]][loc[1]] for mille in cmil for loc in mille]
            loc = tuple(set(cmil[0]) & set(cmil[1]))[0] # intersecting location
            #print(repr(loc))
            if stones_in_cmil.count(me) == 2 and stones_in_cmil.count(opponent) ==0 and board[loc[0]][loc[1]] == self.NONE:
                potential_double_double[me] += 1
            if stones_in_cmil.count(opponent) == 2 and stones_in_cmil.count(me) ==0 and board[loc[0]][loc[1]] == self.NONE:
                potential_double_double[opponent] += 1
        # util structure: a 'min' score or lower will get 0 utility and a 'max' or higher score will get full 'weight'
        # scores between 'min' and 'max' will be trended as follows:
        #    * 'exp'=1 -> 'linear' -- straight line from 0 to 'weight' as measure moves from 'min' to 'max'
        #    * 'exp'=2 -- x squared curve from 0 to 'weight' as measure moves from 'min' to 'max' (slow to ramp near 'min', but strong acceleration at 'max')
        #    * 'exp'=0.5 -- x square-root curve from 0 to 'weight' as measure moves from 'min' to 'max' (quick to ramp near 'min', but slows down at 'max')
        util = { 'libs':            {'min':1, 'max':4, 'weight':.1, 'exp':0.25}, #don't care above 4 but want to heavily weight 0 and 1 liberty
                                                                      #setting max at 4 intrinsically gives preference to locations with
                                                                      #4 neighbors as opposed to locations with 2 or 3.
                 's_count':         {'min':0, 'max':9,  'weight':.2, 'exp':0.25},
                 'dbls':            {'min':0 , 'max':2, 'weight':.15, 'exp':0.5},
                 'milles':          {'min':0 , 'max':1, 'weight':.3, 'exp':1},
                 'alt_milles':      {'min':0 , 'max':1, 'weight':.15, 'exp':1},
                 'poten_dbl_dbl':   {'min':0,  'max':1, 'weight':.1, 'exp':1} }
        score_data = [  {'n':'libs',          'raw_me':liberties[me]+unplaced[me],     'raw_opp':liberties[opponent]+unplaced[opponent]},
                        {'n':'s_count',       'raw_me':stone_count[me],                'raw_opp':stone_count[opponent]},
                        {'n':'dbls',          'raw_me':unblocked_doubles[me],          'raw_opp':unblocked_doubles[opponent]},
                        {'n':'milles',        'raw_me':milles[me],                     'raw_opp':milles[opponent]},
                        {'n':'alt_milles',    'raw_me':alternating_milles[me],         'raw_opp':alternating_milles[opponent]},
                        {'n':'poten_dbl_dbl', 'raw_me':potential_double_double[me],    'raw_opp':potential_double_double[opponent]},
                     ]
        score = 0
        for e in score_data:
            score_delta = 0
            if e['n'] != 'poten_dbl_dbl':
                score_delta += self.normalized_utility(e['raw_me'], util[e['n']])
                score_delta -= self.normalized_utility(e['raw_opp'], util[e['n']])
            else:
                if unplaced[me] >0: score_delta += self.normalized_utility(e['raw_me'], util[e['n']])
                if unplaced[opponent] >0: score_delta -= self.normalized_utility(e['raw_opp'], util[e['n']])
            e['w_score'] = score_delta
            score += score_delta
        if debug and player==0:
            print(' '*10 + 'p={} '.format(player), end='')
            for e in score_data:
                print('{0[n]}:{0[w_score]}({0[raw_me]},{0[raw_opp]}); '.format(e), end='')
            print(' ->{}'.format(score))
##        score += self.normalized_utility(liberties[me]+unplaced[me], util['liberties']) #add unplaced so we don't really use this in the early 'place' game
##        score -= self.normalized_utility(liberties[opponent]+unplaced[opponent], util['liberties'])
##        score += self.normalized_utility(stone_count[me], util['stone count'])
##        score -= self.normalized_utility(stone_count[opponent], util['stone count'])
##        score += self.normalized_utility(unblocked_doubles[me], util['doubles'])
##        score -= self.normalized_utility(unblocked_doubles[opponent], util['doubles'])
##        score += self.normalized_utility(milles[me], util['milles'])
##        score -= self.normalized_utility(milles[opponent], util['milles'])
##        score += self.normalized_utility(alternating_milles[me], util['alt milles'])
##        score -= self.normalized_utility(alternating_milles[opponent], util['alt milles'])
##        if unplaced[me] > 0:
##            score += self.normalized_utility(potential_double_double[me], util['poten_dbl_dbl'])
##        if unplaced[opponent] > 0:
##            score -= self.normalized_utility(potential_double_double[opponent], util['poten_dbl_dbl'])
        #if debug: print('score={}, me={}, stonecount={}, doubles={}, milles={}, alt_mills={}'.format(score, me, stone_count, unblocked_doubles, milles, alternating_milles))
        return round(score, 4)

    def utilities(self, state):
        utils = []
        for player in range(self.n_players):
            utils.append(self.utility(state, player))
        return tuple(utils)

    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        if state.resign is None:
            return False
        else:
            return True

    def num_players(self):
        return self.n_players
    
    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state.player
    
    ##################  ACTION METHODS  ###############################
    def place(self, state, location):
        '''Place a stone at the location indicated and return state.  This is used in phase I
        of the game when each player places stones on the board until they have
        both placed 9 stones.'''
        (me, board, bstones, unplaced, removeok, resign) = state
        assert unplaced[me] > 0
        assert board[location[0]][location[1]] is self.NONE
        unplaced = self.set_tuple(unplaced, me, -1, add=True)
        num = self.set_tuple(bstones, me, 1, add=True)
        new_board = self.set_board(board, location, me)
        if self.creates_mille(self.State(me, new_board, num, unplaced, removeok, resign), me, location):
            return self.State(me, new_board, num, unplaced, True, resign)
        else:
            return self.State(self.other[me], new_board, num, unplaced, removeok, resign)


    def move(self, state, start_location, end_location):
        '''Move a stone from one location to the other and return state.  This is used in
        phase II of the game where players may move their stones one position
        along a line.  NOTE that when a player has 3 stones left, they ma
        place anywhere that is unoccupied on the board, regardless of the
        distance or stones in-between.'''
        (me, board, bstones, unplaced, removeok, resign) = state
        assert board[start_location[0]][start_location[1]] == me
        assert board[end_location[0]][end_location[1]] is self.NONE
        new_board = self.set_board(board, start_location, self.NONE)
        new_board = self.set_board(new_board, end_location, me)
        if self.creates_mille(self.State(me, new_board, bstones, unplaced, removeok, resign), me, end_location):
            return self.State(me, new_board, bstones, unplaced, True, resign)
        else:
            return self.State(self.other[me], new_board, bstones, unplaced, removeok, resign)

    def remove(self, state, location):
        '''Remove a stone from the board and return state (with removeok reset to False).
        This is used in either phase I or II
        after the players gets a Mille (3 in a line) and is called by the
        main game loop after an action sets the 'removeok' flag in the state.
        The player with a mille may remove an opponent's stone but must
        start with opponent stones that do not form a mille of their own.'''
        (me, board, bstones, unplaced, removeok, resign) = state
        assert board[location[0]][location[1]] is self.other[me]
        new_board = self.set_board(board, location, self.NONE)
        num = self.set_tuple(bstones, self.other[me], -1, add=True)
        removeok = False
        return self.State(self.other[me], new_board, num, unplaced, removeok, resign)
        
    def resign(self, state, reason_message):  
        '''Resign'''
        (me, board, bstones, unplaced, removeok, resign) = state
        #Can't modify 'self' (game object) or print anything because this method
        #  is also called in "what if" mode when searching several plies deep for
        #  possible moves.
        return self.State(me, board, bstones, unplaced, removeok, resign=(me, reason_message)) # last element identifies that state.player resigned
    
    ###################  MISC HELPER METHODS  ###############################
    def display(self, state, prev_state=None, prompts=None):
        '''
        Prints out a game board.
        'last_state' gives the state prior to the current state and if present
        will cause an asterix to appear near play locations that changed
        'prompts' is a string of 24 characters (typically 'abcdefghjklmnpqrstuvwxyz')
        which will be used to mark empty intersections instead of '+' so that these
        can be used to prompt a user.
        +---------+---------+
        |         |         |
        |   +-----+-----+   |
        |   |     |     |   |
        |   |  +--+--+  |   |
        |   |  |     |  |   |
        +---+--+     +--+---+
        |   |  |     |  |   |
        |   |  +--+--+  |   |
        |   |     |     |   |
        |   +-----+-----+   |
        |         |         |
        +---------+---------+
        '''
        (me, board, bstones, unplaced, removeok, resign) = state
        board_ch = [] # a copy of the board with None replaced with '+'
        prompt_ch = []
        prev_state = self.last_state
        if prev_state is None:
            last_move = ' '*24
        else:
            last_board = prev_state.board
            last_move = []
        ind_count = 0
        for sq_index, sq in enumerate(board):
            board_ch.append([])
            prompt_ch.append([])
            for v_index, stone in enumerate(sq):
                if stone is self.NONE:
                    board_ch[sq_index].append('+')
                    if prompts:
                        prompt_ch[sq_index].append(prompts[ind_count])
                else:
                    board_ch[sq_index].append(stone)
                    if prompts:
                        prompt_ch[sq_index].append(prompts[ind_count])
                if prev_state:
                    if last_board[sq_index][v_index] == stone:
                        last_move.append(' ') #no change
                    else:
                        last_move.append('*') #change
                ind_count += 1
        if not prompts:
            print('{1[0]}         {1[1]}        {1[2]}'.format(None, last_move))                
            print('{0[0][0]}---------{0[0][1]}---------{0[0][2]}       Player: {1}'.format(board_ch, me))
            print('|   {1[8]}     |{1[9]}    {1[10]}   |       bstones: {0}'.format(bstones, last_move))
            print('|   {0[1][0]}-----{0[1][1]}-----{0[1][2]}   |       Num_unplaced: {1}'.format(board_ch, unplaced))
            print('|   |  {1[16]}  |{1[17]} {1[18]}  |   |       Remove?: {0}'.format(removeok, last_move))
            print('|   |  {0[2][0]}--{0[2][1]}--{0[2][2]}  |   |'.format(board_ch))
            print('|{1[7]}  |{1[15]} |     |{1[17]} |{1[11]} {1[3]}|'.format(None, last_move))
            print('{0[0][7]}---{0[1][7]}--{0[2][7]}{1[23]}   {1[19]}{0[2][3]}--{0[1][3]}---{0[0][3]}'.format(board_ch, last_move))
            print('|   |  |{1[22]} {1[21]} {1[20]}|  |   |'.format(None, last_move))
            print('|   |  {0[2][6]}--{0[2][5]}--{0[2][4]}  |   |'.format(board_ch))
            print('|   |{1[14]}    |{1[13]}    |{1[12]}  |'.format(None, last_move))
            print('|   {0[1][6]}-----{0[1][5]}-----{0[1][4]}   |'.format(board_ch))
            print('|{1[6]}        |{1[5]}       {1[4]}|'.format(None, last_move))
            print('{0[0][6]}---------{0[0][5]}---------{0[0][4]}'.format(board_ch))
            print('')
        else:    
            print('{1[0]}         {1[1]}        {1[2]}'.format(None, last_move)) #row 0

            print('{0[0][0]}---------{0[0][1]}---------{0[0][2]}'.format(board_ch), end='') #row 1
            print('    {0[0][0]}---------{0[0][1]}---------{0[0][2]}'.format(prompt_ch), end='')
            print('    Player: {}'.format(me))

            print('|   {1[8]}     |{1[9]}    {1[10]}   |'.format(None, last_move), end='') #row 2
            print('    |         |         |', end='')
            print('    bstones: {}'.format(bstones))

            print('|   {0[1][0]}-----{0[1][1]}-----{0[1][2]}   |'.format(board_ch), end='') #row 3
            print('    |   {0[1][0]}-----{0[1][1]}-----{0[1][2]}   |'.format(prompt_ch), end='')
            print('    Num_unplaced: {}'.format(unplaced))

            print('|   |  {1[16]}  |{1[17]} {1[18]}  |   |'.format(None, last_move), end='') #row 4
            print('    |   |     |     |   |', end='')
            print('    Remove?: {0}'.format(removeok))

            print('|   |  {0[2][0]}--{0[2][1]}--{0[2][2]}  |   |'.format(board_ch), end='') #row 5
            print('    |   |  {0[2][0]}--{0[2][1]}--{0[2][2]}  |   |'.format(prompt_ch))
            
            print('|{1[7]}  |{1[15]} |     |{1[17]} |{1[11]} {1[3]}|'.format(None, last_move), end='') #row 6
            print('    |   |  |     |  |   |')
            
            print('{0[0][7]}---{0[1][7]}--{0[2][7]}{1[23]}   {1[19]}{0[2][3]}--{0[1][3]}---{0[0][3]}'.format(board_ch, last_move), end='') #row 7
            print('    {0[0][7]}---{0[1][7]}--{0[2][7]}     {0[2][3]}--{0[1][3]}---{0[0][3]}'.format(prompt_ch))

            print('|   |  |{1[22]} {1[21]} {1[20]}|  |   |'.format(None, last_move), end='') #row 8
            print('    |   |  |     |  |   |')

            print('|   |  {0[2][6]}--{0[2][5]}--{0[2][4]}  |   |'.format(board_ch), end='') #row 9
            print('    |   |  {0[2][6]}--{0[2][5]}--{0[2][4]}  |   |'.format(prompt_ch))

            print('|   |{1[14]}    |{1[13]}    |{1[12]}  |'.format(None, last_move), end='') #row 10
            print('    |   |     |     |   |')

            print('|   {0[1][6]}-----{0[1][5]}-----{0[1][4]}   |'.format(board_ch), end='') #row 11
            print('    |   {0[1][6]}-----{0[1][5]}-----{0[1][4]}   |'.format(prompt_ch))

            print('|{1[6]}        |{1[5]}       {1[4]}|'.format(None, last_move), end='') #row 12
            print('    |         |         |')

            print('{0[0][6]}---------{0[0][5]}---------{0[0][4]}'.format(board_ch), end='') #row 13
            print('    {0[0][6]}---------{0[0][5]}---------{0[0][4]}'.format(prompt_ch))
            print('')
        self.last_state = state


    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def creates_mille(self, state, player, location):
        '''Returns True if location is part of a Mille (3 in a row) (assuming that
        location is played with "player" stone, if it is not there already.)'''
        board = state.board
        assert board[location[0]][location[1]] is self.NONE or board[location[0]][location[1]] == player
        for milleset in self.millelist:
            if location in milleset:
                is_mille = True
                for loc in milleset:
                    if loc == location: continue
                    else:
                        loc_value = board[loc[0]][loc[1]]
                        if loc_value is self.NONE or loc_value != player:
                            is_mille = False
                            break
                if is_mille:
                    return True
        return False

    def unblocked_rows(self, state, player, location):
        '''Returns a list with an entry for each row that contains this location
        which does not have an opponent stone.  The entry is 1 if a stone placed
        at location would be the only stone, 2 if a stone placed at the location
        would be part of a pair and a 3 if a stone placed at the location would be
        part of a triple (mille)'''
        board = state.board
        #assert board[location[0]][location[1]] is self.NONE or board[location[0]][location[1]] == player
        unblocked_rows = []
        for row in self.millelist:
            if location in row:
                stones_in_row = []
                for loc in row:
                    if loc == location:
                        stones_in_row.append(player)
                    else:
                        stones_in_row.append(board[loc[0]][loc[1]])
                if self.other[player] not in stones_in_row:
                    unblocked_rows.append(stones_in_row.count(player))
        return unblocked_rows

    def liberties(self, state, location):
        '''Returns the number of empty locations neighboring the current location.
        (Uses the term 'liberty' in the same sense as the word is used for Go).'''
        board = state.board
        liberty_count = 0
        for n in self.neighbors[location]:
            if board[n[0]][n[1]] is self.NONE:
                liberty_count += 1
        return liberty_count

    def player_stones(self, state, player):
        '''Return a list of tuples which are all the locations that
        player has a stone on the board'''
        board = state.board
        stones = []
        for sq in range(len(board)):
            for vertex in range(len(board[sq])):
                if board[sq][vertex] is player:
                    stones.append( (sq, vertex) )
        return stones

    def can_move_to_mille(self, state, location):
        '''Return true if the stone at location can 'move' to a location where it fills a player mille'''
        player = state.board[location[0]][location[1]]
        for n in self.neighbors[location]:
            if state.board[n[0]][n[1]] is self.NONE:
                if 3 in self.unblocked_rows(state, player, n):
                    return True
        return False

    def set_tuple(self, incoming, index, value, add=False):
        '''returns a new tuple that is the same as 'incoming', except
        'index' is set to 'value'.  If add=True, then 'index'
        is replaced with incoming[index]+value.'''
        new_t = [v for v in incoming]
        if add:
            new_t[index] = new_t[index] + value
        else:
            new_t[index] = value
        return tuple(v for v in new_t)

    def set_board(self, board, location, value):
        '''returns a new board with 'location' set to 'value'.
        In need to do this in a function because boards are tuples
        which are (and need to be) immutable'''
        new_board = [ [v for v in inner_list] for inner_list in board]
        new_board[location[0]][location[1]] = value
        return tuple( tuple(v for v in inner_list) for inner_list in new_board)

    def normalized_utility(self, metric, u):
        '''return a normalized utility score.
        'u' is a dictionary with the following keys:
            'min' = metrics at or lower than 'min' should get 0 utility
            'max' = metrics at or higher than 'max' should get 'weight' utility
            'weight' = maximum utility -- normalize actual utility between 0 and 'weight'
            'exp' = exponent used in extrapolation curve, e.g. 1='linear'
            '''
        if metric <= u['min']: return 0
        elif metric >= u['max']: return u['weight']
        else:
            return round(u['weight'] * ((float(metric) - u['min'])/(u['max']-u['min']))**u['exp'],4)
      
def human_player(game, state):
    '''Allow a human player to see the board and select an action'''
    prompt_to_loc = {'a':(0,0), 'b':(0,1), 'c':(0,2), 'd':(0,3), 'e':(0,4),
                     'f':(0,5), 'g':(0,6), 'h':(0,7), 'j':(1,0), 'k':(1,1),
                     'l':(1,2), 'm':(1,3), 'n':(1,4), 'p':(1,5), 'q':(1,6),
                     'r':(1,7), 's':(2,0), 't':(2,1), 'u':(2,2), 'v':(2,3),
                     'w':(2,4), 'x':(2,5), 'y':(2,6), 'z':(2,7)}
               ### DAY TODO - Finish this refactoring      
    game.display(state, prompts='abcdefghjklmnpqrstuvwxyz')
    def query_player(board, action, prompt, valid_check_value, error_prompt):
        '''Prompt the user for a move and return an (action, argument) tuple'''
        print(prompt)
        while True:
            answer = raw_input('Your choice > ')
            try:
                loc = prompt_to_loc[answer]
            except:
                print('Invalid.')
                continue
            if board[loc[0]][loc[1]] == valid_check_value:
                return (action, (loc,))
            else:
                print("Error, must specify a location holding an opponent's stone")
        
    me, board, bstones, unplaced, removeok, resign = state
    if removeok:
        return query_player(board, 'Remove', 'Your action is *Remove*.  Please choose an opponent stone to remove',
                     game.other[me], "Error, must specify a location holding an opponent's stone")

    elif unplaced[me] > 0:
        return query_player(board, 'Place', 'Your action is *Place*. Please choose a location to place a stone',
                            game.NONE, "Error, must specify an empty location (with a letter, not a 0 or 1)")

    else:
        _, loc_tuple = query_player(board, 'Move', 'Your action is *Move*.  Please choose a stone to Move',
                     me, "Error, must specify a location with a {}".format(me))
        start_location = loc_tuple[0]

        _, loc_tuple = query_player(board, 'Move', 'Now select a location to move to',
                     game.NONE, "Error, must specify an empty location (with a letter, not a 0 or 1)")
        end_location = loc_tuple[0]
        return ('Move', (start_location, end_location))




if __name__ == '__main__':
    debug = True
    if not debug: trace = disabled
    
    if False:

        class TestMultiplayerGame(Game):
            def __init__(self, nplayers=3, layers=4, branch=2, rand=True, term_vals=[]):
                self.initial = (0, 0) #initial state  (node, player)
                self.n_players = nplayers
                self.layers = layers
                self.branch_factor = branch
                self.random = rand
                term_values = term_vals
                self.state_map = {}      # {node: (successor_node tuple), }
                self.terminal_nodes = {} # {terminal_node: (utility value tuple), }
                #self.layers, self.branch_factor, self.random, term_values = 4, 2, False, [13,5, 18, 1, 7, 11, 17, 14]
                #self.layers, self.branch_factor, self.random, term_values = 4, 2, False, [[13, 1, 6],[5, 15, 14], [18, 2, 9], [1, 16, 11], [7, 4, 13], [11, 5, 12], [17, 1, 17], [14, 7, 4]]
                #self.layers, self.branch_factor, self.random, term_values = 5, 3, True, []
                node = 0
                random.seed(42)
                for layer in range(self.layers):   # initialize state_map and terminal_nodes.  0 / 1,2,3 / 4,5,6; 7,8,9; 10,11,12 etc.
                    if layer == 0:
                        self.state_map[node] = tuple(x for x in range(1, self.branch_factor+1))
                        node += 1
                    else:
                        for _ in range(self.branch_factor**layer):
                            if layer != self.layers-1:
                                self.state_map[node] = tuple((node*self.branch_factor)+x for x in range(1,self.branch_factor+1))
                                node += 1
                            else: # terminal nodes
                                if self.random:
                                    self.terminal_nodes[node] = [random.randint(1, 20) for x in range(self.num_players())]
                                else:
                                    self.terminal_nodes[node] = term_values.pop(0)
                                node += 1
            def actions(self, state):
                if state[0] in self.terminal_nodes:
                    return []
                else:
                    return self.state_map[state[0]] #these "moves" are really just a tuple of the successor nodes
            def result(self, state, move):
                '''"move" is just a specified successor node'''
                return (move, (state[1]+1)%self.n_players) #new state (node, next_player)
            def utility(self, state, player):
                utilities = self.utilities(state)
                if isinstance(utilities, int):
                    return utilities
                else:
                    return utilities[player]
            def utilities(self, state):
                if state[0] in self.terminal_nodes:
                    return self.terminal_nodes[state[0]]
                else:
                    return (-1,)*self.n_players #junk value for non-terminal nodes just to test
            def terminal_test(self, state):
                return state[0] in self.terminal_nodes
            def num_players(self):
                return self.n_players
            def to_move(self, state):
                return state[1]
            def display(self, state):
                print(self.state_map)
                print(self.terminal_nodes)

        class Fig52Game(Game):
            """The game represented in [Fig. 5.2]. Serves as a simple test case.
            >>> g = Fig52Game()
            >>> minimax_decision('A', g)
            'a1'
            >>> alphabeta_full_search('A', g)
            'a1'
            >>> alphabeta_search('A', g)
            'a1'
            """
            def __init__(self):
                self.succs = dict(A=dict(a1='B', a2='C', a3='D'),
                                 B=dict(b1='B1', b2='B2', b3='B3'),
                                 C=dict(c1='C1', c2='C2', c3='C3'),
                                 D=dict(d1='D1', d2='D2', d3='D3'))
                self.utils = dict(B1=3, B2=12, B3=8, C1=2, C2=4, C3=6, D1=14, D2=5, D3=2)
                self.initial = 'A'

            def actions(self, state):
                return sorted(self.succs.get(state, {}).keys(), reverse=False)

            def result(self, state, move):
                return self.succs[state][move]

            def utility(self, state, player):
                if player == 0: #MAX
                    return self.utils[state]
                else:
                    return -self.utils[state]

            def utilities(self, state):
                return (self.utils[state], -self.utils[state])

            def terminal_test(self, state):
                return state not in ('A', 'B', 'C', 'D')

            def to_move(self, state):
                if state in 'BCD':
                    return 1
                else:
                    return 0
            def num_players(self):
                return 2

        play_game( Fig52Game(), alphabeta_search, alphabeta_multiplayer)

##        play_game( TestMultiplayerGame(nplayers=2, layers=4, branch=2, rand=False, term_vals=[13,5, 18, 1, 7, 11, 17, 14]), alphabeta_search, alphabeta_search)
##        
##        alphabeta_player_d4_t3_multi = functools.partial(alphabeta_player_maker, d=4, t=3, multi=True)
##        play_game( TestMultiplayerGame(nplayers=3, layers=7, branch=2, rand=True,
##                        term_vals=[[13, 1, 6],[5, 15, 14], [18, 2, 9], [1, 16, 11], [7, 4, 13], [11, 5, 12], [17, 1, 17], [14, 7, 4]]),
##                        alphabeta_player_d4_t3_multi, alphabeta_player_d4_t3_multi, alphabeta_player_d4_t3_multi)
##        play_game( TestMultiplayerGame(nplayers=3, layers=4, branch=2, rand=False,
##                        term_vals=[[13, 1, 6],[5, 15, 14], [18, 2, 9], [1, 16, 11], [7, 4, 13], [11, 5, 12], [17, 1, 17], [14, 7, 4]]),
##                        multiplayer_search, multiplayer_search, multiplayer_search)


    if False:
        State = collections.namedtuple('State', ['player', 'board', 'bstones', 'unplaced', 'removeok'])
        N = '_'
        if False:
            test_board = ( (1,1,1,0,0,0,N,1),(N,N,N,N,N,N,N,N),(N,0,N,N,N,N,N,N))
            test_state = State(player=1, board=test_board, bstones=(4,4),
                               unplaced=(4,5), removeok=False)
            #print(alphabeta_player_d1(NineMensMorris(test_state), test_state))
        if False:
            test_board2 = ( (1,N,N,N,N,N,N,N),(N,N,N,N,N,N,N,N),(0,N,N,N,N,N,N,0))
            test_state2 = State(player=1, board=test_board2, bstones=(2,1),
                               unplaced=(7,8), removeok=False)
            game2 = NineMensMorris(test_state2)
            if False:
                game2.display(test_state2)
                game2.utility(test_state2,1)
                after_move_state = game2.result(test_state2,  (game2.place, ((2,6),))  )
                game2.display(after_move_state)
                game2.utility(after_move_state,1)
            actions = game2.actions(test_state2)
            for action in actions:
                game2.display(game2.result(test_state2, action))
                print('{}: --> {}'.format(action, game2.utility(game2.result(test_state2, action),1)))
            #print(alphabeta_player_d1(game2, test_state2))
        if True:
            test_board3 = ( (0,1,1,0,0,1,0,1),(N,N,N,0,N,N,N,N),(N,N,N,1,N,N,N,N))
            test_state3 = State(player=0, board=test_board3, bstones=(5,5),
                               unplaced=(4,4), removeok=False)
            game3 = NineMensMorris(test_state3)
            actions = game3.actions(test_state3)
            for action in actions:
                game3.display(game3.result(test_state3, action))
                print('{}: --> {}'.format(action, game3.utility(game3.result(test_state3, action),0)))
            
    

    if True:
        #alphabeta_player_d6_t3 = functools.partial(alphabeta_player_maker, d=6, t=3)
        #alphabeta_player_d4_t3 = functools.partial(alphabeta_player_maker, d=4, t=3) #dAY, this one against d2 gets in a loop in the move portion of the game -- debug.
        #alphabeta_player_d4_t3_multi = functools.partial(alphabeta_player_maker, d=4, t=3, multi=True)

        
        State = collections.namedtuple('State', ['player', 'board', 'bstones', 'unplaced', 'removeok', 'resign'])
        #test_board = ( (1,3,3,3,3,3,3,3),(3,3,3,3,3,3,3,3),(3,3,3,3,3,3,0,0) )
        #functools.partial(alphabeta_multiplayer, d=4)(NineMensMorris(),
        #    State(player=1, board=test_board, bstones=(2, 1), unplaced=(7, 8), removeok=False, resign=None))

        obvious = State(player=0, board=( (0,0,1,0,3,3,3,3),(3,3,3,3,3,3,3,3),(3,3,3,3,3,3,1,1) ), bstones=(3, 3), unplaced=(6, 6), removeok=False, resign=None)
        print(functools.partial(alphabeta_multiplayer, d=1, alpha_pruning=False)(NineMensMorris(), obvious))
        #print(functools.partial(alphabeta_multiplayer, d=1, alpha_pruning=True )(NineMensMorris(), obvious))

##        player_d4_dual = functools.partial(alphabeta_search, d=4)
##        player_d4_multi = functools.partial(alphabeta_multiplayer, d=4)        
##        print(play_game( NineMensMorris(), player_d4_multi, player_d4_dual))

        #print(play_game( NineMensMorris(), alphabeta_player_d4_t3, alphabeta_player_d2)) #gets in a loop at move
        #print(play_game( NineMensMorris(), random_player, random_player))



###########################################################################################
##  GRAVEYARD
###########################################################################################


