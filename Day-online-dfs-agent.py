import random

debug = False

prev_state = None
prev_action = None
result = {}
untried = {}
unbacktracked = {}

def online_dfs_agent(state):
    global prev_state, prev_action, result, untried, unbacktracked
    if debug: print "-"*20 #debug
    if debug: print "State: ", state, "\n    untried is", untried[state] if state in untried else 'undefined' 
    if debug: print '    unbacktracked is ', unbacktracked[state] if state in unbacktracked else 'undefined'
    #if debug: print '    result is', result
    if is_goal(state):
        return 'stop'
    if state not in untried:
        untried[state] = actions(state)
        if debug: print "added ", untried[state], " to untried"
    if prev_state is not None:
        if (prev_state, prev_action) not in result: #Note: this step not in the text -- it was a bug not to include it or you can loop
            result[(prev_state, prev_action)] = state
            if state not in unbacktracked:
                unbacktracked[state] = []
            unbacktracked[state].append(prev_state)
            if debug: print 'added prev_state ', prev_state, ' to end of unbacktracked for ', state, ':', unbacktracked[state]
    if len(untried[state]) <= 0:
        if len(unbacktracked[state]) <=0:
            return 'stop'
        else:
            if debug: print "BACKTRACKING: using ", unbacktracked[state]
            g = unbacktracked[state].pop()
            for s, a in result.keys():
                if s == state and result[(s,a)] == g:
                    prev_action = a
                    print '    backtrack by', a #debug
    else:
        #prev_action = untried[state].pop(0)
        prev_action = untried[state].pop(random.choice(range(len(untried[state]))))
    prev_state = state
    if debug: print '    RESULT ', result, '\n    UNTRIED ', untried, '\n    UNBACKTRACKED ', unbacktracked #debug
    return prev_action


def actions(state):
    # these hold deltas from current state number
    m = { 1:[3,1], 2:[3,-1,1], 3:[-1, 3], 
          4:[-3], 5:[3,-3], 6:[3,-3], 
          7:[1], 8:[-3,-1], 9:[-3] }
    return m[state]

def is_goal(state):
    return state == 9

def maze_finder():
    state = 1
    while True:
    #for i in range(100): #debug
        action = online_dfs_agent(state)
        print 'at', state, 'moving by', action
        if action == 'stop':
            print 'stop'
            return 
        state += action

maze_finder()
