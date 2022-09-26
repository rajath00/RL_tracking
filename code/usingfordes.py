from collections import namedtuple

#from util import plot_automaton


Transition = namedtuple(typename='Transition', field_names=[
                        'source', 'event', 'target'])


class Automaton(object):

    def __init__(self, states, init, events, trans, marked=None, forbidden=None):
        """
        This is the constructor of the automaton.

        At creation, the automaton gets the following attributes assigned:
        :param states: A set of states
        :param init: The initial state
        :param events: A set of events
        :param trans: A set of transitions
        :param marked: (Optional) A set of marked states
        :param forbidden: (Optional) A set of forbidden states
        """
        self.states = states
        self.init = init
        self.events = events
        self.trans = trans
        self.marked = marked if marked else set()
        self.forbidden = forbidden if forbidden else set()

    def __str__(self):
        """
        Prints the automaton in a pretty way.
        """
        return 'states: \n\t{}\n' \
               'init: \n\t{}\n' \
               'events: \n\t{}\n' \
               'transitions: \n\t{}\n' \
               'marked: \n\t{}\n' \
               'forbidden: \n\t{}\n'.format(
                   self.states, self.init, self.events,
                   '\n\t'.join([str(t) for t in self.trans]), self.marked, self.forbidden)

    def __setattr__(self, name, value):
        """Validates and protects the attributes of the automaton"""
        if name in ('states', 'events'):
            value = frozenset(self._validate_set(value))
        elif name == 'init':
            value = self._validate_init(value)
        elif name == 'trans':
            value = frozenset(self._validate_transitions(value))
        elif name in ('marked', 'forbidden'):
            value = frozenset(self._validate_subset(value))
        super(Automaton, self).__setattr__(name, value)

    def __getattribute__(self, name):
        """Returns a regular set of the accessed attribute"""
        if name in ('states', 'events', 'trans', 'marked', 'forbidden'):
            return set(super(Automaton, self).__getattribute__(name))
        else:
            return super(Automaton, self).__getattribute__(name)

    def __eq__(self, other):
        """Checks if two Automata are the same"""
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def _validate_set(states):
        """Checks that states is a set and the states in it are strings or integers"""
        assert isinstance(states, set)
        for state in states:
            assert isinstance(state, str) or isinstance(
                state, int), 'A state must be either of type string or integer!'
        return states

    def _validate_subset(self, subset):
        """Validates the set and checks whether the states in the subset are part of the state set"""
        subset = self._validate_set(subset)
        assert subset.issubset(
            self.states), 'Marked and forbidden states must be subsets of all states!'
        return subset

    def _validate_init(self, state):
        """Checks whether the state is part of the state set"""
        assert isinstance(state, str) or isinstance(
            state, int), 'The initial state must be of type string or integer!'
        assert state in self.states, 'The initial state must be member of states!'
        return state

    def _validate_transitions(self, transitions):
        """Checks that all transition elements are part in the respective sets (states, events)"""
        assert isinstance(transitions, set)
        for transition in transitions:
            assert isinstance(transition, Transition)
            assert transition.source in self.states
            assert transition.event in self.events
            assert transition.target in self.states
        return transitions

def merge_label(label1, label2):
    """Creates a new label based on two labels"""
    return '{}.{}'.format(label1, label2)

def cross_product(setA, setB):
    """Computes the crossproduct of two sets"""
    return {merge_label(a, b) for b in setB for a in setA}

def filter_trans_by_source(trans, states_to_keep):
    """Returns a new set containing all transitions where the source is in states_to_keep"""
    return {t for t in trans if t.source in states_to_keep}

def filter_trans_by_events(trans, events_to_keep):
    """Returns a new set containing all transitions where the event is in events_to_keep"""
    return {t for t in trans if t.event in events_to_keep}

def filter_trans_by_target(trans, states_to_keep):
    """Returns a new set containing all transitions where the target is in states_to_keep"""
    return {t for t in trans if t.target in states_to_keep}

def extract_elems_from_trans(trans, field):
    """
    Returns a new set with just the elements in a field of all transitions.
    E.g. field='source' for all source states
    or field='event' or field='target'
    """
    return {getattr(t, field) for t in trans}

def flip_trans(trans):
    """ Flips the direction of the transitions in the set"""
    return {Transition(t.target, t.event, t.source) for t in trans}

a1 = Automaton(states={1},
               init=1,
               events=set(),
               trans=set())
a2 = Automaton(states={2},
               init=2,
               events=set(),
               trans=set())

merge_label(a1.states,a2.states)
print(a1.states)


def synch(aut1, aut2):
    """

    Returns the synchronous composition of two automata.



    :param aut1: Automaton

    :param aut2: Automaton

    """

    # YOUR CODE HERE

    events = aut1.events

    events = events.union(set(aut2.events))

    # print("events - ", events)

    init = merge_label(aut1.init, aut2.init)

    # print("initial pos - ", init)

    all_possible_states = cross_product(aut1.states, aut2.states)

    # print("all possible states - ", all_possible_states)

    marked = set()

    if aut1.marked != set() and aut2.marked == set():

        # print("1")

        for i in all_possible_states:

            for j in aut1.marked:

                if str(j) in i:
                    marked.add(i)

elif aut2.marked != set() and aut1.marked == set():

# print("2")

for i in all_possible_states:

    for j in aut2.marked:

        if str(j) in i:
            marked.add(i)

elif aut1.marked != set() and aut2.marked != set():

# print("3")

for i in all_possible_states:

    for j in aut1.marked:

        for k in aut2.marked:

            if j in i and k in i:
                marked.add(i)

forbidden = set()

if aut1.forbidden != set() and aut2.forbidden == set():

    # print("forbidden 1")

    for i in all_possible_states:

        for j in aut1.forbidden:

            if str(j) in i:
                forbidden.add(i)

elif aut1.forbidden == set() and aut2.forbidden != set():

    # print("forbidden 2")

    for i in all_possible_states:

        for j in aut2.forbidden:

            if str(j) in i:
                forbidden.add(i)

else:

    # print("forbidden 3")

    for i in all_possible_states:

        for j in aut1.forbidden:

            if str(j) in i:
                forbidden.add(i)

    for i in all_possible_states:

        for j in aut2.forbidden:

            if str(j) in i:
                forbidden.add(i)

# print("forbidden - ", forbidden)


tran = set()

intersect = (aut1.events).intersection(aut2.events)

if intersect:

    for i in intersect:

        tra1 = filter_trans_by_events(aut1.trans, {i})

        tra2 = filter_trans_by_events(aut2.trans, {i})

        for t1 in tra1:

            for t2 in tra2:
                tran.add(Transition(merge_label(t1.source, t2.source), i, merge_label(t1.target, t2.target)))

if aut1.events - aut2.events:

    for i in aut1.events - aut2.events:

        tra1 = filter_trans_by_events(aut1.trans, {i})

        for t1 in tra1:

            for t2 in aut2.states:
                tran.add(Transition(merge_label(t1.source, t2), i, merge_label(t1.target, t2)))

if aut2.events - aut1.events:

    for j in aut2.events - aut1.events:

        tra2 = filter_trans_by_events(aut2.trans, {j})

        for t1 in aut1.states:

            for t2 in tra2:
                tran.add(Transition(merge_label(t1, t2.source), j, merge_label(t1, t2.target)))

reachable_states = reach(tran, {init})

# print("reachable states -", reachable_states)


main_marked = set()

for i in marked:

    if i in reachable_states:
        main_marked.add(i)

# print("marked -", main_marked)


reachable_transitions = set()

for l in reachable_states:

    filtered = filter_trans_by_source(tran, {l})

    for i in filtered:

        if i.target in reachable_states:
            reachable_transitions.add(i)

# print('reachable transitions -', reachable_transitions)


aut1aut2 = Automaton(states=reachable_states, init=init, events=events, trans=reachable_transitions, marked=main_marked,
                     forbidden=forbidden)

# raise NotImplementedError()

return aut1aut2