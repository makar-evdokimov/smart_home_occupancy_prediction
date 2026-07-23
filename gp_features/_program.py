from copy import copy

import numpy as np
import pandas as pd
from sklearn import metrics




def _select(values, s, e, var='usage_mean'):
    return values[s+24:e+25].T[~nan_indices]

def _sad(a):
    output = np.array(list(map(lambda x: sum(abs(np.diff(x))), a)))
    return(output)

def _range(a):
    output = np.array(list(map(lambda x: max(x)-min(x), a)))
    return(output)

def _mean(a):
    output = np.array(list(map(lambda x: np.mean(x), a)))
    return(output)

def _sd(a):
    output = np.array(list(map(lambda x: np.std(x), a)))
    return(output)

def _max(a):
    output = np.array(list(map(lambda x: max(x), a)))
    return(output)

def _min(a):
    output = np.array(list(map(lambda x: min(x), a)))
    return(output)

def _sum(a):
    output = np.array(list(map(lambda x: np.sum(x), a)))
    return(output)

def _ratio(a1,a2):
    output = np.array(list(map(lambda x, y: x/y, a1, a2)))
    return(output)

def _add(a1,a2):
    output = np.array(list(map(lambda x, y: x+y, a1, a2)))
    return(output)

def _greater(a1,a2):
    output = np.array(list(map(lambda x, y: max(x,y), a1, a2)))
    return(output)

def _lesser(a1,a2):
    output = np.array(list(map(lambda x, y: min(x,y), a1, a2)))
    return(output)


global_functions = {
    '_range':_range,
    '_sad':_sad,
    '_mean':_mean,
    '_sd':_sd,
    '_min':_min,
    '_max':_max,
    '_sum':_sum,
    '_ratio':_ratio,
    '_add':_add,
    '_greater':_greater,
    '_lesser':_lesser,
    '_select':_select,
}



class _Program(object):

    def __init__(self,
                 function_set,
                 arities,
                 df: pd.DataFrame,
                 select_values,
                 init_depth,
                 init_method,
                 p_point_replace,
                 n_features,
                 parsimony_coefficient,
                 first_gen,
                 random_state,
                 feature_names=None,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.df = df
        self.select_values = select_values
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.parsimony_coefficient = parsimony_coefficient
        self.first_gen = first_gen
        self.feature_names = feature_names
        self.program = program

        dict_arities = {}
        for i in range(max(self.arities.values())):
            dict_arities[i + 1] = []
            for function in self.arities:
                if self.arities[function] == i + 1:
                    dict_arities[i + 1].append(function)
        self.dict_arities = dict_arities

        if self.program is not None:
            if not self.validate_program:
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.p_point_replace = p_point_replace

        self.auc_uplift_ = None
        self.parents = None
        self._n_samples = None

    def build_program(self, random_state):

        functions_select = ['mean', 'range', 'sd', 'sad', 'min', 'max', 'sum']
        """Build a naive random program.
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        if self.first_gen:
            size = random_state.uniform()
            func_probs = 0.4
            func_probs = np.cumsum(func_probs)
            func = random_state.uniform()
            if func < func_probs:
                function = 'ratio'
            else:
                other_func = ['lesser', 'greater', 'add']
                function = random_state.randint(len(other_func))
                function = other_func[function]

            program = [function]
            terminal_stack = [self.arities[function]]

            while terminal_stack:
                depth = len(terminal_stack)
                if program[-1] in functions_select:
                    program.append('select')
                    s = random_state.randint(-24, 23)
                    e = random_state.randint(s + 1, 24)
                    program.append((s, e))
                    terminal_stack.append(1)
                choice = self.n_features + len(self.function_set)
                choice = random_state.randint(choice)
                # Determine if we are adding a function or terminal
                depth = len(terminal_stack)
                if ((depth < max_depth) and (
                        method == 'full' or choice <= len(self.function_set)) and (
                        (len(program) < 2) or (program[-2] != 'select'))):
                    function = random_state.randint(len(functions_select))
                    function = functions_select[function]
                    program.append(function)
                    terminal_stack.append(self.arities[function])
                    depth = len(terminal_stack)
                else:
                    terminal = random_state.randint(self.n_features)
                    program.append(terminal)
                    terminal_stack[-1] -= 1
                    while terminal_stack[-1] == 0:
                        terminal_stack.pop()
                        if not terminal_stack:
                            return program
                        terminal_stack[-1] -= 1

            # We should never get here
            return None

        else:
            # Start a program with a function to avoid degenerative programs
            function = random_state.randint(len(self.function_set))
            function = self.function_set[function]
            program = [function]
            terminal_stack = [self.arities[function]]

            while terminal_stack:
                depth = len(terminal_stack)
                if program[-1] in functions_select:
                    program.append('select')
                    s = random_state.randint(-24, 23)
                    e = random_state.randint(s + 1, 24)
                    program.append((s, e))
                    terminal_stack.append(1)
                choice = self.n_features + len(self.function_set)
                choice = random_state.randint(choice)
                # Determine if we are adding a function or terminal
                depth = len(terminal_stack)
                if ((depth < max_depth) and (
                        method == 'full' or choice <= len(self.function_set)) and (
                        (len(program) < 2) or (program[-2] != 'select'))):
                    function = random_state.randint(len(self.function_set))
                    function = self.function_set[function]
                    program.append(function)
                    terminal_stack.append(self.arities[function])
                    depth = len(terminal_stack)
                else:
                    terminal = random_state.randint(self.n_features)
                    program.append(terminal)
                    terminal_stack[-1] -= 1
                    while terminal_stack[-1] == 0:
                        terminal_stack.pop()
                        if not terminal_stack:
                            return program
                        terminal_stack[-1] -= 1

            # We should never get here
            return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        functions_select = {'mean', 'range', 'sd', 'sad', 'min', 'max', 'sum'}
        for node in self.program:
            if node in functions_select:
                terminals.append(3)
            elif node in self.function_set:
                terminals.append(self.arities[node])
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if node in self.function_set:
                terminals.append(self.arities[node])
                output += node + '('
            elif node == 'select':
                output += node + '('
            elif isinstance(node, tuple):
                output += f"{node[0]},{node[1]},"
            elif isinstance(node, int):
                if self.feature_names is None:
                    output += 'X%s' % node
                else:
                    output += f"\'{self.feature_names[node]}\'"
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if isinstance(self.program[i - 1], tuple):
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.
        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.
        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.
        """

        graph_function_names = {
            'range': 'range',
            'sad': 'sad',
            'mean': 'mean',
            'sd': 'sd',
            'min': 'min',
            'max': 'max',
            'sum': 'sum',
            'ratio': 'ratio',
            'add': 'sum',
            'greater': 'max',
            'lesser': 'min',
        }
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if node in self.function_set:
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([self.arities[node], i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, graph_function_names[node], fill))
            elif node == 'select':
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([1, i])
                output += (f'{i} [label=\"{node}{self.program[i + 1]}\", fillcolor=\"{fill}\"] ;\n')
            elif isinstance(node, tuple):
                continue
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if self.feature_names is None:
                    feature_name = 'X%s' % node
                else:
                    feature_name = self.feature_names[node]
                if isinstance(self.program[i - 1], tuple):
                    i -= 1
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, feature_name, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][-1],
                                                terminals[-1][1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if node in self.function_set:
                terminals.append(self.arities[node])
                depth = max(len(terminals), depth)
            elif node == 'select':
                terminals.append(1)
                depth = max(len(terminals), depth)
            elif isinstance(node, tuple):
                continue
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        length = len(self.program)
        for node in self.program:
            if node == 'select':
                length -= 2
        return length

    def execute(self):
        """Execute the program according to X.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """
        # Check for single-node programs
        print(f"\nThe program being estimated is {self.__str__()}")
        X = self.df[self.feature_names].values
        node = self.program[0]
        if isinstance(node, int):
            if self.n_features == 1:
                output = X
            else:
                output = X[:, node]
            return output

        global_functions = {
            'range': _range,
            'sad': _sad,
            'mean': _mean,
            'sd': _sd,
            'min': _min,
            'max': _max,
            'sum': _sum,
            'ratio': _ratio,
            'add': _add,
            'greater': _greater,
            'lesser': _lesser,
        }

        functions_select = ['mean', 'range', 'sd', 'sad', 'min', 'max', 'sum']

        apply_stack = []

        for i, node in enumerate(self.program):

            if node in self.function_set:
                apply_stack.append([node])
            elif (not isinstance(self.program[i - 1], tuple)) and (isinstance(node, int)):
                apply_stack.append(node)
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

        intermediate_values = []

        while apply_stack:
            if isinstance(apply_stack[-1], int):
                intermediate_result = apply_stack[-1]
            else:
                function_name = apply_stack[-1][0]
                function = global_functions[function_name]
                arity = self.arities[function_name]
                if function_name in functions_select:
                    intermediate_result = function(_select(
                        self.select_values, apply_stack[-1][2][0],
                        apply_stack[-1][2][1], self.feature_names[apply_stack[-1][3]]
                    ))
                else:
                    # Apply functions that have sufficient arguments

                    if self.n_features == 1:
                        terminals = [X if isinstance(t, int)
                                     else t for t in intermediate_values[:arity]]

                    else:
                        terminals = [X[:, t] if isinstance(t, int)
                                     else t for t in intermediate_values[:arity]]
                    intermediate_result = function(*terminals)
                    del intermediate_values[:arity]

            if len(apply_stack) != 1:
                intermediate_values.insert(0, intermediate_result)
                apply_stack.pop()
            else:
                return intermediate_result

        return None

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.
        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.
        """
        if program is None:
            program = self.program
            length = self.length_
        else:
            length = len(program)
            for node in program:
                if node == 'select':
                    length -= 2

        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.

        probs = np.array([0.9 if (node in self.function_set)
                          else 0 if (node == 'select') | isinstance(node, tuple)
        else 0.1 for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if length > 3:
                no_full_tree_cond = (node == program[0])
            else:
                no_full_tree_cond = False
            if node in self.function_set:
                stack += self.arities[node]
            elif (node == 'select') | isinstance(node, tuple):
                stack += 1
            end += 1
            if isinstance(program[start - 1], tuple) | no_full_tree_cond:
                start = np.searchsorted(probs, random_state.uniform())
                stack = 1
                end = start

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.
        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.
        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.
        random_state : RandomState instance
            The random number generator.
        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.
        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.
        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.
        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if program[node] in self.function_set:
                arity = self.arities[program[node]]
                # Find a valid replacement with same arity
                replacement = len(self.dict_arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.dict_arities[arity][replacement]
                program[node] = replacement
            elif program[node] == 'select':
                continue
            elif isinstance(program[node], tuple):
                s = random_state.randint(-24, 23)
                e = random_state.randint(s + 1, 24)
                program[node] = (s, e)

            else:
                terminal = random_state.randint(self.n_features)
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)

    def fitness(self, baseline_value, y, baseline_features, train_ix, test_ix, model, verbose=True):
        X = np.append(self.df[baseline_features].values,
                          np.reshape(self.execute(), (-1, 1)), axis=1)
        model.fit(X[train_ix], y[train_ix], verbose=-1)
        auc = metrics.roc_auc_score(y[test_ix], model.predict_proba(X[test_ix])[:, 1])
        raw_fitness = auc - baseline_value
        penalty = self.parsimony_coefficient * self._length()
        fitness = raw_fitness - penalty
        if verbose:
            print("The penalized value of its fitness is {}".format(fitness))
        self.fitness_ = fitness
        return (fitness)