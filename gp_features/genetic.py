import itertools
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
import joblib

from ._program import _Program

from .utils import _partition_estimators
from .utils import check_random_state
from .utils import house_split_indices


def _parallel_evolve(n_programs, parents, X, y, seeds, params, model, baseline_features,
                     baseline_value, train_ix, test_ix):
    """Private function used to build a batch of programs within a job."""

    if len(X.shape) == 1:
        n_features = 1
        n_samples = X.shape[0]
    else:
        n_samples, n_features = X.shape

    # Unpack parameters
    tournament_size = params['tournament_size']
    df = params['df']
    select_values = params['select_values']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    parsimony_coefficient = params['parsimony_coefficient']
    feature_names = params['feature_names']

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        parent_index = contenders[np.argmax(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            first_gen = True
            genome = None
        else:
            first_gen = False

            method = random_state.uniform()
            parent, parent_index = _tournament()

            if parent.length_ == 2:
                if method < 0.8:
                    # point_mutation
                    program, mutated = parent.point_mutation(random_state)
                    genome = {'method': 'Point Mutation',
                              'parent_idx': parent_index,
                              'parent_nodes': mutated}
                else:
                    # reproduction
                    program = parent.reproduce()
                    genome = {'method': 'Reproduction',
                              'parent_idx': parent_index,
                              'parent_nodes': []}

            else:
                if method < method_probs[0]:
                    # crossover
                    donor, donor_index = _tournament()
                    program, removed, remains = parent.crossover(donor.program,
                                                                 random_state)
                    genome = {'method': 'Crossover',
                              'parent_idx': parent_index,
                              'parent_nodes': removed,
                              'donor_idx': donor_index,
                              'donor_nodes': remains}

                elif method < method_probs[1]:
                    # subtree_mutation
                    program, removed, _ = parent.subtree_mutation(random_state)
                    genome = {'method': 'Subtree Mutation',
                              'parent_idx': parent_index,
                              'parent_nodes': removed}
                elif method < method_probs[2]:
                    # hoist_mutation
                    program, removed = parent.hoist_mutation(random_state)
                    genome = {'method': 'Hoist Mutation',
                              'parent_idx': parent_index,
                              'parent_nodes': removed}
                elif method < method_probs[3]:
                    # point_mutation
                    program, mutated = parent.point_mutation(random_state)
                    genome = {'method': 'Point Mutation',
                              'parent_idx': parent_index,
                              'parent_nodes': mutated}
                else:
                    # reproduction
                    program = parent.reproduce()
                    genome = {'method': 'Reproduction',
                              'parent_idx': parent_index,
                              'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           df=df,
                           select_values=select_values,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           first_gen=first_gen,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        programs.append(program)

    return programs


def evolution(X=None, y=None, df=None, select_values=None,
              baseline_features=None, train_fraction=0.7,
              model=None,
              population_size=100,
              hall_of_fame=100,
              generations=1000,
              tournament_size=5,
              init_depth=(3, 3),
              init_method='half and half',
              function_set=['mean', 'range', 'sd',
                            'sad', 'min', 'max',
                            'sum', 'ratio', 'add',
                            'greater', 'lesser'],
              arities={
                  'ratio': 2,
                  'mean': 1,
                  'range': 1,
                  'sd': 1,
                  'sad': 1,
                  'min': 1,
                  'max': 1,
                  'sum': 1,
                  'add': 2,
                  'greater': 2,
                  'lesser': 2
              },
              p_crossover=0.8,
              p_subtree_mutation=0.05,
              p_hoist_mutation=0,
              p_point_mutation=0.1,
              p_point_replace=0.33,
              parsimony_coefficient=0.0001,
              feature_names=None,
              n_jobs=1,
              random_state=None,
              warm_start=False,
              low_memory=False,
              verbose=0,
              programs=None):
    def verbose_reporter(run_details=None):
        """A report of the progress of the evolution process.
        Parameters
        ----------
        run_details : dict
            Information about the evolution.
        """

        # Estimate remaining time for run
        gen = run_details['generation'][-1]
        generation_time = run_details['generation_time'][-1]
        if generation_time > 60:
            generation_time = '{0:.2f}m'.format(generation_time / 60.0)
        else:
            generation_time = '{0:.2f}s'.format(generation_time)

        print(f"\nGeneration {gen} is over.\
        \nThe average length of the program is {run_details['average_length'][-1]}\
        \nThe average fitness value is {run_details['average_fitness'][-1]}\
        \nThe length of the best program is {run_details['best_length'][-1]}\
        \nThe fitness value of the best program is {run_details['best_fitness'][-1]}\
        \nIt took " + generation_time + " to run the generation\n")

    random_state = check_random_state(random_state)

    if len(X.shape) == 1:
        n_features = 1
    else:
        _, n_features_ = X.shape

    if hall_of_fame is None:
        hall_of_fame = population_size

    method_probs = np.array([p_crossover,
                             p_subtree_mutation,
                             p_hoist_mutation,
                             p_point_mutation])
    method_probs = np.cumsum(method_probs)

    params = {}
    params['tournament_size'] = tournament_size
    params['df'] = df
    params['select_values'] = select_values
    params['arities'] = arities
    params['init_depth'] = init_depth
    params['init_method'] = init_method
    params['method_probs'] = method_probs
    params['p_point_replace'] = p_point_replace
    params['parsimony_coefficient'] = parsimony_coefficient
    params['feature_names'] = feature_names
    params['function_set'] = function_set

    if not warm_start or programs is None:
        # Free allocated memory, if any
        programs = []
        run_details = {'generation': [],
                       'average_length': [],
                       'average_fitness': [],
                       'best_length': [],
                       'best_fitness': [],
                       'generation_time': []}

    prior_generations = len(programs)
    n_more_generations = generations - prior_generations

    if warm_start:
        # Generate and discard seeds that would have been produced on the
        # initial fit call.
        for i in range(len(programs)):
            _ = random_state.randint(np.iinfo(np.int32).max, size=population_size)

    cv_index = list(df['cv_index'].unique())

    for gen in range(prior_generations, generations):

        start_time = time()

        if gen == 0:
            parents = None
        else:
            parents = programs[gen - 1]

        train_ix, test_ix = house_split_indices(df, cv_index, train_fraction, verbose=1)
        model.fit(df.iloc[train_ix, :][baseline_features], y[train_ix])

        baseline_value = metrics.roc_auc_score(
            y[test_ix], model.predict_proba(df.iloc[test_ix, :][baseline_features])[:, 1]
        )

        print(f"\nThe value of AUC for the baseline model in Generation {gen} is {baseline_value}")

        # Parallel loop
        n_jobs, n_programs, starts = _partition_estimators(
            population_size, n_jobs)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=population_size)

        population = Parallel(n_jobs=n_jobs,
                              verbose=int(verbose > 1))(
            delayed(_parallel_evolve)(n_programs[i],
                                      parents,
                                      X,
                                      y,
                                      seeds[starts[i]:starts[i + 1]],
                                      params,
                                      model,
                                      baseline_features,
                                      baseline_value, train_ix, test_ix)
            for i in range(n_jobs))

        # Reduce, maintaining order across different n_jobs
        population = list(itertools.chain.from_iterable(population))

        fitness = [program.fitness(baseline_value, y, baseline_features, train_ix, test_ix, model) for program in
                   population]

        length = [program.length_ for program in population]

        programs.append(population)

        if (gen + 1) % 100 == 0:
            joblib.dump(programs, f"programs_{gen + 1}gens.pkl")

        # Remove old programs that didn't make it into the new population.
        if not low_memory:
            for old_gen in np.arange(gen, 0, -1):
                indices = []
                for program in programs[old_gen]:
                    if program is not None:
                        for idx in program.parents:
                            if 'idx' in idx:
                                indices.append(program.parents[idx])
                indices = set(indices)
                for idx in range(population_size):
                    if idx not in indices:
                        programs[old_gen - 1][idx] = None
        elif gen > 0:
            # Remove old generations
            programs[gen - 1] = None

        best_program = population[np.argmax(fitness)]

        run_details['generation'].append(gen)
        run_details['average_length'].append(np.mean(length))
        run_details['average_fitness'].append(np.mean(fitness))
        run_details['best_length'].append(best_program.length_)
        run_details['best_fitness'].append(best_program.fitness(
            baseline_value, y,
            baseline_features, train_ix, test_ix,
            model, verbose=False
        ))
        generation_time = time() - start_time
        run_details['generation_time'].append(generation_time)

        if verbose:
            verbose_reporter(run_details)

    # Find the best individuals in the final generation
    fitness = np.array(fitness)

    _hall_of_fame = fitness.argsort()[::-1][:hall_of_fame]
    best_programs = [programs[-1][i] for i in
                     _hall_of_fame]

    return best_programs
