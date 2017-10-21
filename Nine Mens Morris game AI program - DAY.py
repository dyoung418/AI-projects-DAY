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
import random
import functools
import itertools
import collections

################################################################################
## Create memoizer decorator to speed up lookup of states that have
## already been calculated.
def decorator(d):
    "Make function d a decorator: d wraps a function fn."
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

################################################################################
## This defines the game of Nine Men's Morris -- the "what"

other = {1:0, 0:1}
State = collections.namedtuple('State', ['player', 'board', 'num_stones', 'unplaced_stones', 'remove_earned'])
initial_state = State(player=0, board=tuple( tuple(None for _ in range(8)) for _ in range(3) ),
                      num_stones=(0,0), unplaced_stones=(9,9), remove_earned=False)
loss_condition = 2 # 2 or fewer stones == lose
last_state = initial_state

##### ACTIONS  ####
def place(state, location):
    '''Place a stone at the location indicated and return state.  This is used in phase I
    of the game when each player places stones on the board until they have
    both placed 9 stones.'''
    (me, board, num_stones, unplaced_stones, remove_earned) = state
    assert unplaced_stones[me] > 0
    assert board[location[0]][location[1]] is None
    unplaced = set_tuple(unplaced_stones, me, -1, add=True)
    num = set_tuple(num_stones, me, 1, add=True)
    new_board = set_board(board, location, me)
    if creates_mille(State(me, new_board, num, unplaced, remove_earned), me, location):
        return State(me, new_board, num, unplaced, True)
    else:
        return State(other[me], new_board, num, unplaced, remove_earned)

def move(state, start_location, end_location):
    '''Move a stone from one location to the other and return state.  This is used in
    phase II of the game where players may move their stones one position
    along a line.  NOTE that when a player has 3 stones left, they may
    place anywhere that is unoccupied on the board, regardless of the
    distance or stones in-between.'''
    (me, board, num_stones, unplaced_stones, remove_earned) = state
    assert board[start_location[0]][start_location[1]] == me
    assert board[end_location[0]][end_location[1]] is None
    new_board = set_board(board, start_location, None)
    new_board = set_board(new_board, end_location, me)
    if creates_mille(State(me, new_board, num_stones, unplaced_stones, remove_earned), me, end_location):
        return State(me, new_board, num_stones, unplaced_stones, True)
    else:
        return State(other[me], new_board, num_stones, unplaced_stones, remove_earned)

def remove(state, location):
    '''Remove a stone from the board and return state (with remove_earned reset to False).
    This is used in either phase I or II
    after the players gets a Mille (3 in a line) and is called by the
    main game loop after an action sets the 'remove_earned' flag in the state.
    The player with a mille may remove an opponent's stone but must
    start with opponent stones that do not form a mille of their own.'''
    (me, board, num_stones, unplaced_stones, remove_earned) = state
    assert board[location[0]][location[1]] is other[me]
    print('**** REMOVE **** -- player {} removes {}'.format(me, location))
    new_board = set_board(board, location, None)
    num = set_tuple(num_stones, other[me], -1, add=True)
    remove_earned = False
    return State(other[me], new_board, num, unplaced_stones, remove_earned)
    
def resign_because_trapped(state):
    '''Resign because no move steps are possible in Phase II'''
    pass #won't be called but documenting it here as one of the actions

##### MAIN PROGRAM ######
def play_ninemensmorris(A, B, verbose=False):
    '''Play a game of Nine Men's Morris using strategy A for player 1 and strategy B
    for player 2.
    A and B strategies are functions that return an action (place, move) and a tuple
    for the arguments to that action
    Return value is a tuple of (strategy_function_that_won, winning_player, end_state)'''

    strategies = [A, B] #these will be functions that return an action and tuple of args to that action
    state = initial_state
    global last_state
    
    while True:
        if verbose: print_state(state, prev_state=last_state) #DEBUG
        (me, board, num_stones, unplaced_stones, remove_earned) = state
        you = other[me]
        if num_stones[you] + unplaced_stones[you] <= loss_condition:
            print('WIN -- YOU RAN OUT OF STONES')
            return strategies[me], me, state
        elif num_stones[me] + unplaced_stones[me] <= loss_condition:
            print('LOSE -- I RAN OUT OF STONES')
            return strategies[you], you, state
        else:
            action, args = strategies[me](state)
            if action == 'place':
                if verbose: print('    place -- player {} places at {}'.format(state.player, args[0]))
                state = place(state, *args)
                (_, _, _, _, remove_earned) = state
                if remove_earned:
                    action, args = strategies[me](state)
                    assert action == 'remove'
                    state = remove(state, *args)
            elif action == 'move':
                if verbose: print('    move -- player {} moves {}->{}'.format(state.player, args[0], args[1]))
                state = move(state, *args)
                (_, _, _, _, remove_earned) = state
                if remove_earned:
                    action, args = strategies[me](state)
                    assert action == 'remove'
                    state = remove(state, *args)
            elif action == 'resign_because_trapped':
                print('LOSE -- I AM TRAPPED')
                return strategies[other[me]], other[me], state
            else: # Illegal action?  You lose!
                print('LOSE -- I MADE AN ILLEGAL MOVE = {}'.format(action))
                return strategies[other[me]], other[me], state

################################################################################
## This defines strategies for playing Mille -- the "how"

def lazy_strategy(state):
    '''A strategy that could probably never win, unless a player 
    makes an illegal move'''
    me, board, num_stones, unplaced_stones, remove_earned = state
    if remove_earned:
        opponent_stones = player_stones(state, other[me])
        is_mille_stone = functools.partial(creates_mille, state, other[me])
        eligible_stones = list(itertools.ifilterfalse(is_mille_stone, opponent_stones))
        if eligible_stones is None or len(eligible_stones) < 1:
            eligible_stones = opponent_stones
        #print('eligible stones: {}'.format(eligible_stones)) #DEBUG
        #print('random choice: {}'.format(random.choice(eligible_stones))) #DEBUG
        return ('remove', ( random.choice(eligible_stones), ))
    elif unplaced_stones[me] > 0: #Phase 1: need to place
        action = 'place'
        for sq in range(len(board)):
            for vertex in range(len(board[sq])):
                if board[sq][vertex] is None:
                    return (action, ((sq,vertex),)) #just take first blank spot
    else:
        action = 'move'
        for stone in player_stones(state, me):
            for n in Neighbors[stone]:
                if board[n[0]][n[1]] is None:
                    return (action, (stone, n))
                    
def lazy_strategy2(state):
    return lazy_strategy(state)

def set_strategy(state):
    '''A strategy that could probably never win, unless a player 
    makes an illegal move'''
    me, board, num_stones, unplaced_stones, remove_earned = state
    if remove_earned:
        opponent_stones = player_stones(state, other[me])
        is_mille_stone = functools.partial(creates_mille, state, other[me])
        eligible_stones = list(itertools.ifilterfalse(is_mille_stone, opponent_stones))
        if eligible_stones is None or len(eligible_stones) < 1:
            eligible_stones = opponent_stones
        #print('eligible stones: {}'.format(eligible_stones)) #DEBUG
        #print('random choice: {}'.format(random.choice(eligible_stones))) #DEBUG
        return ('remove', ( eligible_stones[-1], ))
    elif unplaced_stones[me] > 0: #Phase 1: need to place
        action = 'place'
        first_places = [ (2,3), (2,1), (0,2), (1,6), (0,4), (2,0), (2,2), (1,3)]
        for loc in first_places:
            if board[loc[0]][loc[1]] is None:
                return (action, (loc,))
        for sq in range(len(board)):
            for vertex in range(len(board[sq])):
                if board[sq][vertex] is None:
                    return (action, ((sq,vertex),)) #just take first blank spot
    else:
        action = 'move'
        for stone in player_stones(state, me):
            for n in Neighbors[stone]:
                if board[n[0]][n[1]] is None:
                    return (action, (stone, n))

def optimal_strategy(state):
    '''Keeping this here since it reflects the structure for game
    strategy that completely traverses all possibilities, but for
    Nine Men's Morris, this approach will not be feasible'''
    return best_action(state, all_possible_actions, Q_mille, U_mille)
    
def limited_depth_search_strategy(state):
    #DAY TODO
    pass
                
def planner_strategy(state):
    #DAY TODO
    pass

def utility_strategy(state, random_mistakes=0, index_pick=0):
    '''Choose an action based on utilities of those *actions* (not the 
    resulting states as in the Q/U framework below).
    'random_mistakes' says that X percent of the time, you don't choose the
    action that you normally would have by calling this function.
    'index_pick' of 0 says pick the zero'th element in a sorted list
    of actions (sorted by highest utility) -- so it means pick the highest
    utility.  index_pick=1 would mean pick the second-highest utility.'''
    all_actions = all_possible_actions(state)
    index = index_pick
    groups = []
    scores = []
    sort_function = functools.partial(score_action, state)
    sorted_actions = sorted(all_actions, key=sort_function, reverse=True)
    for k, g in itertools.groupby(sorted_actions, sort_function):
        groups.append(list(g))
        scores.append(k)
    #print scores, groups  #DEBUG
    if random_mistakes > 0:
        if random.random() <= random_mistakes:
            index = index_pick + random.choice([1,2])
    if len(groups) == 0:
        return ('resign_because_trapped', None)
    else:
        return random.choice(groups[min(index, len(groups)-1)])

def human_player(state):
    '''Allow a human player to see the board and select an action'''
    prompt_to_loc = {'a':(0,0), 'b':(0,1), 'c':(0,2), 'd':(0,3), 'e':(0,4),
                     'f':(0,5), 'g':(0,6), 'h':(0,7), 'j':(1,0), 'k':(1,1),
                     'l':(1,2), 'm':(1,3), 'n':(1,4), 'p':(1,5), 'q':(1,6),
                     'r':(1,7), 's':(2,0), 't':(2,1), 'u':(2,2), 'v':(2,3),
                     'w':(2,4), 'x':(2,5), 'y':(2,6), 'z':(2,7)}
                     
    print_state(state, prev_state=last_state, prompts='abcdefghjklmnpqrstuvwxyz')
    me, board, num_stones, unplaced_stones, remove_earned = state
    if remove_earned:
        action = 'remove'
        print('Your action is *{}*\nPlease choose an opponent stone to remove:'.format(action))
        while True:
            answer = raw_input('Your choice > ')
            try:
                loc = prompt_to_loc[answer]
            except:
                print('Invalid.')
                continue
            if board[loc[0]][loc[1]] == other[me]:
                return (action, (loc,))
            else:
                print("Error, must specify a location holding an opponent's stone")
    elif unplaced_stones[me] > 0:
        action = 'place'
        print('Your action is *{}*\nPlease choose a location to place a stone'.format(action))
        while True:
            answer = raw_input('Your choice > ')
            try:
                loc = prompt_to_loc[answer]
            except:
                print('Invalid')
                continue
            if board[loc[0]][loc[1]] is None:
                return (action, (loc,))
            else:
                print("Error, must specify an empty location (with a letter, not a 0 or 1)")
    else:
        action = 'move'
        print('Your action is *{}*\nPlease choose a stone to move:'.format(action))
        while True:
            answer = raw_input('Your choice > ')
            try:
                loc = prompt_to_loc[answer]
            except:
                print('Invalid')
                continue
            if board[loc[0]][loc[1]] == me:
                start_location = loc
                break
            else:
                print("Error, must specify a location with a {}".format(me))
        print('Now select a location to move to')
        while True:
            answer = raw_input('Your choice > ')
            try:
                loc = prompt_to_loc[answer]
            except:
                print('Invalid')
                continue
            if board[loc[0]][loc[1]] is None:
                return (action, (start_location, loc))
            else:
                print("Error, must specify an empty location (with a letter, not a 0 or 1)")
        
################################################################################
## This defines functions for dealing with probability trees -- the "how"

# This first function is a playing strategy (and therefore also belongs in the
# the section above) but it is generalized and can be used in just about any
# problem like this.
def best_action(state, actions, Q, U):
    """Return the optimal action for a state, given U (utility
    function: U(state) = score between 1 and 0.), and
    Q (quality function: Q(state, action, U) = expected utility of taking action
    from initial state)"""
    def EU(action): return Q(state, action, U)
    return max(actions(state), key = EU)

action_dict = {'remove': remove, 'move': move, 'place':place}

def Q_mille(state, action, U):
    """Quality function: The expected utility of choosing 'action' from initial 'state'.
    Here 'U' is the utility function
    Note that for Mille, 'action' is actually a tuple of (action, *args), but those
    *args don't include 'state' which every action needs as its first argument."""
    #1. get the action function with action_dict[action[0]]
    #2. call the action with arguments 'state' plus the arguments given in action[1]
    #3. return the utility (from U()) of the resulting state
    return U(action_dict[action[0]](state, *action[1]))

@memo
def U_mille(state): #Utility function 
    """The utility of a state, given as a value from 0 to 1;  Note that the
    utility given is the utility for the player specified in the state.
    Note that since the Utility function calls the Quality function and vica versa,
    these end up being recursive, so it is important that this Utility function has
    states where it exits to end the recursion.  In this case it is when either me or you
    has won."""
    # Assumes opponent also plays with this strategy.
    me, board, num_stones, unplaced_stones, remove_earned = state
    you = other[me]
    if num_stones[you] + unplaced_stones[you] <= loss_condition:
        return 1 # I won -- maximum utility
    elif num_stones[me] + unplaced_stones[me] <= loss_condition:
        return 0 # I lost -- no utility
    else:
        return max(Q_mille(state, action, U_mille) # I will choose the action that has the highest U
                   for action in all_possible_actions(state))



################################################################################
## Some helper subfunctions

@memo
def all_possible_actions(state):
    '''Return a list of all possible actions from the given state.
    Note that this assumes that the player given in the state has
    the next turn.
    If state has 'remove_earned', then the action must be 'remove', but
    there may still be multiple args for 'remove'.'''
    (me, board, num_stones, unplaced_stones, remove_earned) = state
    if remove_earned: action = 'remove'
    elif unplaced_stones[me] > 0: action = 'place'
    else: action = 'move'
    if action == 'remove':
        opponent_stones = player_stones(state, other[me])
        is_mille_stone = functools.partial(creates_mille, state, other[me])
        eligible_stones = list(itertools.ifilterfalse(is_mille_stone, opponent_stones))
        if eligible_stones is None or len(eligible_stones) < 1:
            eligible_stones = opponent_stones #there is debate on whether it is OK to remove Mille stones if that is all that is available: this implementation says yes, it is.
        return [ ( action, (location,) ) for location in eligible_stones ]
    elif action == 'place':
        empty_locs = player_stones(state, None)
        return [ ( action, (location,) ) for location in empty_locs ]
    elif action == 'move':
        return_actions = []
        for start_stone in player_stones(state, me):
            for n in Neighbors[start_stone]:
                if board[n[0]][n[1]] is None:
                    return_actions.append( (action, (start_stone, n)) )
        return return_actions

def set_tuple(incoming, index, value, add=False):
    '''returns a new tuple that is the same as 'incoming', except
    'location' is set to 'value'.  If add=True, then 'location'
    is replaced with location+value.'''
    new_t = [v for v in incoming]
    if add:
        new_t[index] = new_t[index] + value
    else:
        new_t[index] = value
    return tuple(v for v in new_t)

def set_board(board, location, value):
    '''returns a new board with 'location' set to 'value'.
    In need to do this in a function because boards are tuples
    which are (and need to be) immutable'''
    new_board = [ [v for v in inner_list] for inner_list in board]
    new_board[location[0]][location[1]] = value
    return tuple( tuple(v for v in inner_list) for inner_list in new_board)

@memo
def player_stones(state, player):
    '''Return a list of tuples which are all the locations that
    player has a stone on the board'''
    (_, board, _, _, _) = state
    stones = []
    for sq in range(len(board)):
        for vertex in range(len(board[sq])):
            if board[sq][vertex] is player:
                stones.append( (sq, vertex) )
    return stones

##@memo
##def neighbors(state, location):
##    '''return a list of tuples which represent the locations of the
##    board neighbors of the given location'''
##    (me, board, num_stones, unplaced_stones, remove_earned) = state
##    num_squares = len(board) #generallizing in case I want to nest more squares
##    num_vertices = len(board[0]) #although I'm generalizing here, it will always be 8
##    sq, vertex = location
##    sq_neighbors = []
##    if vertex == 0:
##        sq_neighbors += [(sq,1), (sq,num_vertices-1)]
##    elif vertex == num_vertices -1:
##        sq_neighbors += [(sq,0), (sq, vertex-1)]
##    else:
##        sq_neighbors += [(sq,vertex+1), (sq, vertex-1)]
##    if vertex % 2 == 1: # a middle vertex
##        if sq == 0: # large outer square
##            return sq_neighbors + [(sq+1, vertex)]
##        elif sq == num_squares -1: # smallest inner square
##            return sq_neighbors + [(sq-1, vertex)]
##        else: # middle square
##            return sq_neighbors + [(sq+1, vertex), (sq-1, vertex)]
##    else: # a corner vertex
##        return sq_neighbors

        
def initiate_Neighbors(state):
    '''return a dictionary which has keys of every location on the board and
    values as a list of tuples which are that location's neighbors.'''
    neighbors = dict()
    (_, board, _, _, _) = state
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
Neighbors = initiate_Neighbors(initial_state)

def initiate_Milles(state):
    '''Return a dictionary which has keys of every location on the board and
    values as a list of all the Mille locations that form a trio for a Mille
    with the key location.  The value is a list of lists, each inner list
    being a list of tuple locations.  (Note that this is subtley different
    than Neighbors where the value is simply a list of location tuples).
    This is for checking if a stone just placed at 'location' forms a mille
    since you can use 'location' as the key to this dict and get the other
    locations that you must check'''
    milles = dict()
    (_, board, _, _, _) = state
    sqs = len(board)
    vs = len(board[0])
    for s in range(sqs):
        for v in range(vs):
            if v % 2 == 0: #corner vertex
                milles[(s,v)] = [ [(s,(v+1)%8), (s,(v+2)%8)], [(s,(v-1)%8), (s,(v-2)%8)] ]
            else: #middle vertex
                milles[(s,v)] = [ [(s,(v+1)%8), (s,(v-1)%8)], [((s+1)%sqs,v), ((s-1)%sqs,v)] ]
    return milles
Milles = initiate_Milles(initial_state)

def initiate_MilleList(state):
    '''Return a list of all triplet locations which can form milles.
    This can be used for analyzing a board, for example, to count
    how many 2-in-a-row sets there are in a configuration.  You'd want
    to do such a thing only within the possible sets of milles.'''
    milles = list()
    (_, board, _, _, _) = state
    sqs = len(board)
    vs = len(board[0])
    for s in range(sqs):
        for v in range(vs):
            if v % 2 == 0: #corner vertex
                milles.append([(s,v), (s,(v+1)%8), (s,(v+2)%8)])
            elif s == 0: #middle vertex and the outer square
                milles.append([( (s+i)%8, v) for i in range(sqs)])
    return milles
MilleList = initiate_MilleList(initial_state)
    
@memo
def creates_mille(state, player, location):
    '''Returns True if location is part of a Mille (3 in a row) (assuming that
    location is played with "player" stone, if it is not there already.)'''
    (_, board, _, _, _) = state
    assert board[location[0]][location[1]] is None or board[location[0]][location[1]] == player
    mille_possibilities = Milles[location]
    for mp in mille_possibilities:
        is_mille = True
        for location in mp:
            loc_value = board[location[0]][location[1]]
            if loc_value is None or loc_value != player:
                is_mille = False
                break
        if is_mille:
            return True
    return False

##@memo
##def part_of_x_unblocked_pairs(state, player, location):
##    '''Returns a num>=1 if the a 'player' played at location would be part of a
##    pair of stones plus a blank, such that the group could potentially be a 
##    mille in the following play.  Returns 0 if the location is not part of such a pair.  The number
##    returned is the number of unblocked pairs that the location is part of.
##    
##    By returning the number instead of True/False, this can also be used to
##    test whether the location is part of a "fork" where a mille can be 
##    completed in one of two locations, both involving this location.'''
##    (_, board, _, _, _) = state
##    assert board[location[0]][location[1]] is None or board[location[0]][location[1]] == player
##    mille_possibilities = Milles[location]
##    stones_in_mille = set()
##    unblocked_pair_count = 0
##    for mp in mille_possibilities:
##        for location in mp:
##            stones_in_mille.add(board[location[0]][location[1]])
##        if stones_in_mille >= set(player, None) and other[player] not in stones_in_mille:
##            unblocked_pair_count += 1
##    return unblocked_pair_count
    
@memo
def unblocked_rows(state, player, location):
    '''Returns a list with an entry for each row that contains this location
    which does not have an opponent stone.  The entry is 1 if a stone placed
    at location would be the only stone, 2 if a stone placed at the location
    would be part of a pair and a 3 if a stone placed at the location would be
    part of a triple (mille)'''
    (_, board, _, _, _) = state
    #assert board[location[0]][location[1]] is None or board[location[0]][location[1]] == player
    mille_possibilities = Milles[location]
    unblocked_rows = [] #return value -- will be list of numbers
    for mp in mille_possibilities:
        stones_in_row = []
        for location in mp:
            stones_in_row.append(board[location[0]][location[1]])
        if other[player] not in stones_in_row:
            unblocked_rows.append(stones_in_row.count(player)+1)
    return unblocked_rows

@memo
def liberties(state, location):
    '''Returns the number of empty locations neighboring the current location.
    (Uses the term 'liberty' in the same sense as the word is used for Go).'''
    (_, board, _, _, _) = state
    liberty_count = 0
    for n in Neighbors[location]:
        if board[n[0]][n[1]] is None:
            liberty_count += 1
    return liberty_count

@memo
def score_action(state, action):
    '''Returns a score (roughly between 0 and 2 -- higher is better).
    This will be added up from a number of utility scores for different
    attributes including the following:
    1. How many liberties the location has
    2. Whether the location could form a pair or a mille for the player
    3. Whether the location could block a pair or mille for the opposing player
    4. Whether the location could form a fork (double mille threat) for the player or opponent'''
    (me, _, _, _, _) = state    
    place_utilities = {
        'liberties': [0, 0.01, 0.02, 0.03, 0.04],
        'player_pairs': [0, 0.4, 0.5], #[zero pairs, one pair, two pairs which is same as a fork]
        'player_mille': 1,
        'opponent_pairs': [0, 0, 0.3],
        'opponent_mille': 0.9,
        'unoccupied_row': [0, 0.05, 0.08],
    }
    remove_utilities = {
        'opponent_pairs': [0, 0.1, 0.3],
        'opponent_mille': 0.9,
    }
    action_type, args = action

    if action_type == 'place' or action_type == 'move':
        if action_type == 'move':
            state = move(state, *args)
            p, b, n, u, r = state # need to force player in after_state to me (it might be other[me] after the 'move')
            state = State(me, b, n, u, False)
            location = args[-1]
        else:
            location = args[0]
        score = 0
        player_unblocked_rows = unblocked_rows(state, me, location)
        opponent_unblocked_rows = unblocked_rows(state, other[me], location)
        score += place_utilities['liberties'][liberties(state, location)]
        score += place_utilities['player_mille'] * (3 in player_unblocked_rows)
        score += place_utilities['opponent_mille'] * (3 in opponent_unblocked_rows)
        score += place_utilities['player_pairs'][player_unblocked_rows.count(2)]
        score += place_utilities['opponent_pairs'][opponent_unblocked_rows.count(2)]
        score += place_utilities['unoccupied_row'][player_unblocked_rows.count(0)]
        return score
    elif action_type == 'remove':
        score = 0
        location = args[0]
        opponent_unblocked_rows = unblocked_rows(state, other[me], location)
        score += remove_utilities['opponent_mille'] * (3 in opponent_unblocked_rows)
        score += remove_utilities['opponent_pairs'][opponent_unblocked_rows.count(2)]
        #DAY TODO -- add 'one move away from a mille' criteria?
        #  for example, if I am blocking an opponent mille (and they have a stone
        #  ready to move into that mille, then I wouldn't move out of that blocking
        #  position -- but I currently have nothing preventing me from doing so.
        return score

    
def print_state(state, prev_state=None, prompts=None):
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
    global last_state
    (me, board, num_stones, unplaced_stones, remove_earned) = state
    board_ch = [] # a copy of the board with None replaced with '+'
    prompt_ch = []
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
            if stone is None:
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
        print('|   {1[8]}     |{1[9]}    {1[10]}   |       Num_stones: {0}'.format(num_stones, last_move))
        print('|   {0[1][0]}-----{0[1][1]}-----{0[1][2]}   |       Num_unplaced: {1}'.format(board_ch, unplaced_stones))
        print('|   |  {1[16]}  |{1[17]} {1[18]}  |   |       Remove?: {0}'.format(remove_earned, last_move))
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
        print('    Num_stones: {}'.format(num_stones))

        print('|   {0[1][0]}-----{0[1][1]}-----{0[1][2]}   |'.format(board_ch), end='') #row 3
        print('    |   {0[1][0]}-----{0[1][1]}-----{0[1][2]}   |'.format(prompt_ch), end='')
        print('    Num_unplaced: {}'.format(unplaced_stones))

        print('|   |  {1[16]}  |{1[17]} {1[18]}  |   |'.format(None, last_move), end='') #row 4
        print('    |   |     |     |   |', end='')
        print('    Remove?: {0}'.format(remove_earned))

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
    last_state = state
                


if __name__ == '__main__':
    #win_strategy, state = play_ninemensmorris(smart_strategy, set_strategy)
    
    best_util = functools.partial(utility_strategy, index_pick=0)
    best_util.__name__ = 'best_util'
    second_best_util = functools.partial(utility_strategy, index_pick=1)
    second_best_util.__name__ = 'second_best_util'
    best_util_20 = functools.partial(utility_strategy, random_mistakes=0.2)
    winner, win_player, state = play_ninemensmorris(human_player, best_util)
    print(winner.__name__, win_player)
    
    
