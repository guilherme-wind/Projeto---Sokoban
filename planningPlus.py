"""Planning (Chapters 10-11)"""
"""Adaptado por Helena Aidos"""

import itertools
import copy
import numpy as np
from logic import FolKB, conjuncts, conjunctsSet, unify_mm, associate
from utils import Expr, expr, first
import search

class PlanningProblem:

    """
    Planning Domain Definition Language (PlanningProblem) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, initial, goals, actions, domain=None):
        self.initial = self.convert(initial) if domain is None else self.convert(initial) + self.convert(domain)
        self.initial.sort()
        self.goals = self.convert(goals)
        self.actions = actions
        self.domain = domain

    def convert(self, clauses):
        """Converts strings into exprs"""
        if not isinstance(clauses, Expr):
            if len(clauses) > 0:
                clauses = expr(clauses)
            else:
                clauses = []
        try:
            clauses = conjuncts(clauses)
        except AttributeError:
            pass

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def expand_fluents(self, name=None):

        kb = None
        if self.domain:
            kb = FolKB(self.convert(self.domain))
            for action in self.actions:
                if action.precond:
                    for fests in set(action.precond).union(action.effect).difference(self.convert(action.domain)):
                        if fests.op[:3] != 'Not':
                            kb.tell(expr(str(action.domain) + ' ==> ' + str(fests)))

        objects = set(arg for clause in set(self.initial + self.goals) for arg in clause.args)
        fluent_list = []
        if name is not None:
            for fluent in self.initial + self.goals:
                if str(fluent) == name:
                    fluent_list.append(fluent)
                    break
        else:
            fluent_list = list(map(lambda fluent: Expr(fluent[0], *fluent[1]),
                                   {fluent.op: fluent.args for fluent in self.initial + self.goals +
                                    [clause for action in self.actions for clause in action.effect if
                                     clause.op[:3] != 'Not']}.items()))

        expansions = []
        for fluent in fluent_list:
            for permutation in itertools.permutations(objects, len(fluent.args)):
                new_fluent = Expr(fluent.op, *permutation)
                if (self.domain and kb.ask(new_fluent) is not False) or not self.domain:
                    expansions.append(new_fluent)

        return expansions

    def expand_actions_old(self, name=None, verbose=False):
        """Generate all possible actions with variable bindings for precondition selection heuristic.
           When name is None all actions are generated, otherwise all name actions will be generated with variable bindings."""
        
        # It is called only once!!!!!!!

        has_domains = all(action.domain for action in self.actions if action.precond)
        if verbose:
            print('all actions with preconditions have domains?',has_domains)
        kb = None
        if has_domains:
            kb = FolKB(self.initial)
            for action in self.actions:
                if action.precond:
                    print(expr(str(action.domain) + ' ==> ' + str(action)))
                    kb.tell(expr(str(action.domain) + ' ==> ' + str(action)))

        objects = set(arg for clause in self.initial for arg in clause.args)
        if verbose:
            print('Objectos:',objects)  
            # Estes objectos são todas as entidades que aparecem no estado inicial, mas não são todas as entidades ui ui
        expansions = []
        action_list = []
        if name is not None:       # name or not name it is the question (all actions or a subset of them, with name 
            for action in self.actions:
                if str(action.name) == name:
                    action_list.append(action)     # may we have more than an action with the same name?
                    break
        else:
            action_list = self.actions
        if verbose:
            print('Generating concrete actions:')
        for action in action_list:
            if verbose:
                print('scheme=',action,'with args=',action.args)
            if action.args==(): ### ou a acção não tem vars
                expansions.append(action)
                if verbose:
                    print('The propositional action',action,'is added')
            else:
                permutations = itertools.permutations(objects, len(action.args))
                # permutations never assign the same value to tuples, like(A,A), 
                # and If need it in the formulation I shall make a new special schema?
                for permutation in permutations:
                    if verbose:
                        print('remember, the action scheme is',action, 'and permutation=',permutation)
                    bindings = unify_mm(Expr(action.name, *action.args), Expr(action.name, *permutation))
                    if verbose:
                        print('bindings=',bindings,'-empty set?',bindings==dict())
                    if bindings != dict():
                        new_args = []
                        for arg in action.args:
                            if arg in bindings:
                                new_args.append(bindings[arg])
                            else:
                                new_args.append(arg)
                        new_expr = Expr(str(action.name), *new_args)
                        if verbose:
                            print('after propagating bindings in the action header=',new_expr)
                        # three cases:
                        #     1. All actions with preconditions have domains and it is true this new expression
                        #     2. All actions with preconditions have domains but this action has no preconditions
                        #     3. Not all actions with preconditions have domains
                        if (has_domains and kb.ask(new_expr) is not False) or (
                                has_domains and not action.precond) or not has_domains:
                            if verbose:
                                print('Lets propagate those bindings towards the preconditions')
                            new_preconds = []
                            if verbose:
                                print('Preconds:::::',action.precond)
                            for precond in action.precond:
                                new_precond_args = []
                                for arg in precond.args:
                                    if arg in bindings:
                                        new_precond_args.append(bindings[arg])
                                    else:
                                        new_precond_args.append(arg)
                                new_precond = Expr(str(precond.op), *new_precond_args)
                                new_preconds.append(new_precond)
                            new_effects = []
                            for effect in action.effect:
                                new_effect_args = []
                                for arg in effect.args:
                                    if arg in bindings:
                                        new_effect_args.append(bindings[arg])
                                    else:
                                        new_effect_args.append(arg)
                                new_effect = Expr(str(effect.op), *new_effect_args)
                                new_effects.append(new_effect)
                            if verbose:
                                print('effects:',new_effects)
                                print('In fact we have created a new concrete action:',(new_expr,new_preconds,new_effects))
                            expansions.append(Action(new_expr, new_preconds, new_effects))
                        elif verbose:
                            print('action not applicable!!!!!!!')
        return expansions
    
    # try to see where the domains are added to the preconds of actions!!!!!!
    def expand_actions(self, name=None, verbose=False):
        """Generate all possible actions with variable bindings for precondition selection heuristic.
           When name is None all actions are generated, otherwise all name actions will be generated with variable bindings."""
        
        # It is called only once!!!!!!!

        has_domains = all(action.domain for action in self.actions if action.precond)
        has_domains=True
        #if verbose:
        #    print('all actions with preconditions have domains?',has_domains)
        #kb = None  might be improved .... if any action has domains???????
        kb = FolKB(self.initial)  # criamos uma kb com base no initial (no estado)
        for action in self.actions:
            if action.precond and action.domain:  # why precond and domain not both empty????
                if verbose:
                    print(expr(str(action.domain) + ' ==> ' + str(action)))
                kb.tell(expr(str(action.domain).replace("~","Not") + ' ==> ' + str(action)))
                # gero uma implicação: se domínio então action para depois tentar provar o action depois de unificar. Pois
        objects = set(arg for clause in self.initial for arg in clause.args)
        objects = set(o for o in objects if o.args==())
        if verbose:
            print('Objectos:',objects)  
            # Estes objectos são todas as entidades que aparecem no estado inicial, mas não são todas as entidades ui ui
        expansions = []
        action_list = []
        if name is not None:       # name or not name it is the question (all actions or a subset of them, with name 
            for action in self.actions:
                if str(action.name) == name:
                    action_list.append(action)     # may we have more than an action with the same name?
                    break
        else:
            action_list = self.actions
        if verbose:
            print('\nGenerating concrete actions:')
        for action in action_list:
            if verbose:
                print('\nSCHEME =',action,'with args =',action.args)
            if action.args==(): ### ou a acção não tem vars
                expansions.append(action)
                if verbose:
                    print('The propositional action',action,'is added')
            else:
                permutations = itertools.permutations(objects, len(action.args))
                # permutations of all objects never assign a repeated value to tuples, like(A,A), 
                # and If need it in the formulation I shall make a new special schema?
                for permutation in permutations:
                    if verbose:
                        print('\nRemember, the action scheme is',action, 'and permutation =',permutation)
                    bindings = unify_mm(Expr(action.name, *action.args), Expr(action.name, *permutation))
                    if verbose:
                        print('bindings =',bindings,'-empty set?',bindings==dict())
                    if bindings != dict():
                        new_args = []
                        for arg in action.args:
                            if arg in bindings:
                                new_args.append(bindings[arg])
                            else:
                                new_args.append(arg)
                        new_expr = Expr(str(action.name), *new_args)
                        if verbose:
                            print('after propagating bindings in the action header =',new_expr)
                        # three cases:
                        #     1. All actions with preconditions have domains and it is true this new expression
                        #     2. All actions with preconditions have domains but this action has no preconditions
                        #     3. Not all actions with preconditions have domains
                        if (action.domain and kb.ask(new_expr) is not False) or not action.domain:
                            if verbose:
                                print('Lets propagate those bindings towards the preconditions')
                            new_preconds = []
                            for precond in action.precond:
                                new_precond_args = []
                                for arg in precond.args:
                                    if arg in bindings:
                                        new_precond_args.append(bindings[arg])
                                    else:
                                        new_precond_args.append(arg)
                                new_precond = Expr(str(precond.op), *new_precond_args)
                                new_preconds.append(new_precond)
                            if verbose:
                                print('preconds:',new_preconds)
                                print('and also propagate them to the effects')
                            new_effects = []
                            for effect in action.effect:
                                new_effect_args = []
                                for arg in effect.args:
                                    if arg in bindings:
                                        new_effect_args.append(bindings[arg])
                                    else:
                                        new_effect_args.append(arg)
                                new_effect = Expr(str(effect.op), *new_effect_args)
                                new_effects.append(new_effect)
                            if verbose:
                                print('effects:',new_effects)
                                print('In fact we have created a new concrete action:',(new_expr,new_preconds,new_effects))
                            expansions.append(Action(new_expr, new_preconds, new_effects))
                        elif verbose:
                            print('domains not satisfied - action not applicable!!!!!!!')
        return expansions

    def is_strips(self):
        """
        Returns True if the problem does not contain negative literals in preconditions and goals
        """
        return (all(clause.op[:3] != 'Not' for clause in self.goals) and
                all(clause.op[:3] != 'Not' for action in self.actions for clause in action.precond))

    def goal_test(self):
        """Checks if the goals have been reached"""
        return all(goal in self.initial for goal in self.goals)

    def act(self, action):
        """
        Performs the action given as argument the action Expre (not the real action)
        If there is more than one action with the same action name, picks the first one.
        The preconditions are tested before the action is performed, only if they are satisfied.
        Note that action is an Expr like expr('Remove(Glass, Table)') or expr('Eat(Sandwich)')
        """
        action_name = action.op
        args = action.args
        list_action = first(a for a in self.actions if a.name == action_name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action_name))
        if not list_action.check_precond(self.initial, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.initial = list_action(self.initial, args).clauses


class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PlanningProblem.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negative literals.
    Negative preconditions and effects are defined by adding a 'Not' before the name of the clause
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("NotEaten(food)")]
    effect = [expr("Eaten(food)"), expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, action, precond, effect, domain=None):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond) if domain is None else self.convert(precond) + self.convert(domain)
        self.effect = self.convert(effect)
        self.domain = domain

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}'.format(Expr(self.name, *self.args))

    def convert(self, clauses):
        """Converts strings into Exprs"""
        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                if clauses[i].op == '~':
                    clauses[i] = expr('Not' + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            clauses = clauses.replace('~', 'Not')
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def relaxed(self):
        """
        Removes delete list from the action by removing all negative literals from action's effect
        """
        return Action(Expr(self.name, *self.args), self.precond,
                      list(filter(lambda effect: effect.op[:3] != 'Not', self.effect)))

    def substitute(self, e, args):
        """Replaces variables in expression with their respective Propositional symbol"""

        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i, _ in enumerate(self.args):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """Checks if the precondition is satisfied in the current state"""

        if isinstance(kb, list):
            kb = FolKB(kb)
        for clause in self.precond:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's knowledge base"""
        if isinstance(kb, list):
            kb = FolKB(kb)

        if not self.check_precond(kb, args):
            raise Exception('Action pre-conditions not satisfied')
        for clause in self.effect:
            inst_clause=self.substitute(clause, args)
            if inst_clause not in kb.clauses:  # evita duplicados
                kb.tell(inst_clause)
            if clause.op[:3] == 'Not':  # Se for uma negação que indica retirar
                new_clause = Expr(clause.op[3:], *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
            else:
                new_clause = Expr('Not' + clause.op, *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
        kb.clauses.sort()
        return kb    
    
    
    
    
    

class ForwardPlanSet(search.Problem):
    """
    [Section 10.2.1]
    Forward state-space search
    """

    def __init__(self, planning_problem,display=False,verbose=False):
        super().__init__(associate('&', planning_problem.initial), associate('&', planning_problem.goals))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions(verbose=verbose)
        if display:
            print('As',len(self.expanded_actions),'Ações expandidas:',self.expanded_actions)

    def actions(self, state):
        return [action for action in self.expanded_actions if all(pre in conjunctsSet(state) for pre in action.precond)]

    #  Tenho de ver o que faz o associate por causa de saber se dois estados parecidos, com as conjunções trocadas
    #  são ou não realmente considerados como iguais. É importante por causa da procura em grafo. OLÉ
    #  Também é importante saber o que faz o action (pode ser este a ordenar
    def result(self, state, action):
        return associate('&', action(conjuncts(state), action.args).clauses)

    def goal_test(self, state):
        return all(goal in conjuncts(state) for goal in self.planning_problem.goals)

    def h(self, state):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=state.state,
                                                   goals=self.goal,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf

class ForwardPlan(search.Problem):
    """
    [Section 10.2.1]
    Forward state-space search
    """

    def __init__(self, planning_problem,display=False,verbose=False):
        super().__init__(associate('&', planning_problem.initial), associate('&', planning_problem.goals))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions(verbose=verbose)
        if display:
            print('As',len(self.expanded_actions),'Ações expandidas:',self.expanded_actions)

    def actions(self, state):
        return [action for action in self.expanded_actions if all(pre in conjuncts(state) for pre in action.precond)]

    #  Tenho de ver o que faz o associate por causa de saber se dois estados parecidos, com as conjunções trocadas
    #  são ou não realmente considerados como iguais. É importante por causa da procura em grafo. OLÉ
    #  Também é importante saber o que faz o action (pode ser este a ordenar
    def result(self, state, action):
        return associate('&', action(conjuncts(state), action.args).clauses)

    def goal_test(self, state):
        return all(goal in conjuncts(state) for goal in self.planning_problem.goals)

    def h(self, state):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=state.state,
                                                   goals=self.goal,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf

class BackwardPlan(search.Problem):
    """
    [Section 10.2.2]
    Backward relevant-states search
    """

    def __init__(self, planning_problem,display=False):
        super().__init__(associate('&', planning_problem.goals), associate('&', planning_problem.initial))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions()
        if display:
            print('N. de ações:',len(self.expanded_actions))
            print('Ações expandidas:',self.expanded_actions)

    def actions(self, subgoal, verbose=False):
        """
        Returns True if the action is relevant to the subgoal, i.e.:
        - the action achieves an element of the effects
        - the action doesn't delete something that needs to be achieved
        - the preconditions are consistent with other subgoals that need to be achieved
        """

        def negate_clause(clause):
            return Expr(clause.op.replace('Not', ''), *clause.args) if clause.op[:3] == 'Not' else Expr(
                'Not' + clause.op, *clause.args)

        subgoal = conjuncts(subgoal)  # deveria retirar las cosas del dominio que no mudam!!!!!!!!!
        
        if verbose:
            print('F1: Filtrar acções tais que pelo menos um dos seus efeitos esteja nos sub-objectivos')
            print([action for action in self.expanded_actions if any(prop in action.effect for prop in subgoal)])
            print()
            print('F2: Filtra as acções em que nenhum dos seus efeitos corresponde à negação de um dos sub-objectivos')
            print([action for action in self.expanded_actions if not any(negate_clause(prop) in subgoal for prop in action.effect)])
            print()
            print('F3: Filtrar ações em que todas as pré-condições da acção correspondentes a sub-objectivos negados têm estar negadas nos efeitos')
            print([action for action in self.expanded_actions 
                   if not any(negate_clause(prop) in subgoal and negate_clause(prop) not in action.effect 
                              for prop in action.precond)])
            print()
            
        return [action for action in self.expanded_actions if
                (any(prop in action.effect for prop in subgoal) and
                 not any(negate_clause(prop) in subgoal for prop in action.effect) and
                 not any(negate_clause(prop) in subgoal and negate_clause(prop) not in action.effect
                         for prop in action.precond))]
    
      ### sacamos os sub-objectivos (aos quais deveríamos remover os elementos do domínio)
      ### queremos as acções tais que:
      ### alguns dos seus efeitos esteja nos sub-objectivos e
      ### nenhum dos seus efeitos aparece negado nos sub-objectivos e
      ### nenhuma das negações das suas pre-condições, pode ser um dos sub-objectivos e simultaneamente não estar nos efeitos) 

    def result(self, subgoal, action, verbose=False):
       
        # g' = (g - effects(a)) + preconds(a)
        if verbose:
            print('Sub-goal:',set(conjuncts(subgoal)))
            print('Action:', action)
            print('\tEffect:',action.effect)
            print('\tPrecond:',action.precond)
            
            print()
                    
        return associate('&', set(set(conjuncts(subgoal)).difference(action.effect)).union(action.precond))

    def goal_test(self, subgoal):
        return all(goal in conjuncts(self.goal) for goal in conjuncts(subgoal))

    def h(self, subgoal):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=self.goal,
                                                   goals=subgoal.state,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf
            
            
class Level:
    """
    Contains the state of the planning problem
    and exhaustive list of actions which use the
    states as pre-condition.
    """

    def __init__(self, kb):
        """Initializes variables to hold state and action details of a level"""

        self.kb = kb
        # current state
        self.current_state = kb.clauses
        # current action to state link
        self.current_action_links = {}
        # current state to action link
        self.current_state_links = {}
        # current action to next state link
        self.next_action_links = {}
        # next state to current action link
        self.next_state_links = {}
        # mutually exclusive actions
        self.mutex = []

    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()

    def separate(self, e):
        """Separates an iterable of elements into positive and negative parts"""

        positive = []
        negative = []
        for clause in e:
            if clause.op[:3] == 'Not':
                negative.append(clause)
            else:
                positive.append(clause)
        return positive, negative

    def find_mutex(self):
        """Finds mutually exclusive actions"""

        # Inconsistent effects
        pos_nsl, neg_nsl = self.separate(self.next_state_links)

        for negeff in neg_nsl:
            new_negeff = Expr(negeff.op[3:], *negeff.args)
            for poseff in pos_nsl:
                if new_negeff == poseff:
                    for a in self.next_state_links[poseff]:
                        for b in self.next_state_links[negeff]:
                            if {a, b} not in self.mutex:
                                self.mutex.append({a, b})

        # Interference will be calculated with the last step
        pos_csl, neg_csl = self.separate(self.current_state_links)

        # Competing needs
        for pos_precond in pos_csl:
            for neg_precond in neg_csl:
                new_neg_precond = Expr(neg_precond.op[3:], *neg_precond.args)
                if new_neg_precond == pos_precond:
                    for a in self.current_state_links[pos_precond]:
                        for b in self.current_state_links[neg_precond]:
                            if {a, b} not in self.mutex:
                                self.mutex.append({a, b})

        # Inconsistent support
        state_mutex = []
        for pair in self.mutex:
            next_state_0 = self.next_action_links[list(pair)[0]]
            if len(pair) == 2:
                next_state_1 = self.next_action_links[list(pair)[1]]
            else:
                next_state_1 = self.next_action_links[list(pair)[0]]
            if (len(next_state_0) == 1) and (len(next_state_1) == 1):
                state_mutex.append({next_state_0[0], next_state_1[0]})

        self.mutex = self.mutex + state_mutex

    def build(self, actions, objects):
        """Populates the lists and dictionaries containing the state action dependencies"""

        for clause in self.current_state:
            p_expr = Expr('P' + clause.op, *clause.args)
            self.current_action_links[p_expr] = [clause]
            self.next_action_links[p_expr] = [clause]
            self.current_state_links[clause] = [p_expr]
            self.next_state_links[clause] = [p_expr]

        for a in actions:
            num_args = len(a.args)
            possible_args = tuple(itertools.permutations(objects, num_args))

            for arg in possible_args:
                if a.check_precond(self.kb, arg):
                    for num, symbol in enumerate(a.args):
                        if not symbol.op.islower():
                            arg = list(arg)
                            arg[num] = symbol
                            arg = tuple(arg)

                    new_action = a.substitute(Expr(a.name, *a.args), arg)
                    self.current_action_links[new_action] = []

                    for clause in a.precond:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links[new_action].append(new_clause)
                        if new_clause in self.current_state_links:
                            self.current_state_links[new_clause].append(new_action)
                        else:
                            self.current_state_links[new_clause] = [new_action]

                    self.next_action_links[new_action] = []
                    for clause in a.effect:
                        new_clause = a.substitute(clause, arg)

                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links:
                            self.next_state_links[new_clause].append(new_action)
                        else:
                            self.next_state_links[new_clause] = [new_action]

    def perform_actions(self):
        """Performs the necessary actions and returns a new Level"""

        new_kb = FolKB(list(set(self.next_state_links.keys())))
        return Level(new_kb)


class Graph:
    """
    Contains levels of state and actions
    Used in graph planning algorithm to extract a solution
    """

    def __init__(self, planning_problem):
        self.planning_problem = planning_problem
        self.kb = FolKB(planning_problem.initial)
        self.levels = [Level(self.kb)]
        self.objects = set(arg for clause in self.kb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def expand_graph(self):
        """Expands the graph by a level"""

        last_level = self.levels[-1]
        last_level(self.planning_problem.actions, self.objects)
        self.levels.append(last_level.perform_actions())

    def non_mutex_goals(self, goals, index):
        """Checks whether the goals are mutually exclusive"""

        goal_perm = itertools.combinations(goals, 2)
        for g in goal_perm:
            if set(g) in self.levels[index].mutex:
                return False
        return True


class GraphPlan:
    """
    Class for formulation GraphPlan algorithm
    Constructs a graph of state and action space
    Returns solution for the planning problem
    """

    def __init__(self, planning_problem):
        self.graph = Graph(planning_problem)
        self.no_goods = []
        self.solution = []

    def check_leveloff(self):
        """Checks if the graph has levelled off"""

        check = (set(self.graph.levels[-1].current_state) == set(self.graph.levels[-2].current_state))

        if check:
            return True

    def extract_solution(self, goals, index):
        """Extracts the solution"""

        level = self.graph.levels[index]
        if not self.graph.non_mutex_goals(goals, index):
            self.no_goods.append((level, goals))
            return

        level = self.graph.levels[index - 1]

        # Create all combinations of actions that satisfy the goal
        actions = []
        for goal in goals:
            actions.append(level.next_state_links[goal])

        all_actions = list(itertools.product(*actions))

        # Filter out non-mutex actions
        non_mutex_actions = []
        for action_tuple in all_actions:
            action_pairs = itertools.combinations(list(set(action_tuple)), 2)
            non_mutex_actions.append(list(set(action_tuple)))
            for pair in action_pairs:
                if set(pair) in level.mutex:
                    non_mutex_actions.pop(-1)
                    break

        # Recursion
        for action_list in non_mutex_actions:
            if [action_list, index] not in self.solution:
                self.solution.append([action_list, index])

                new_goals = []
                for act in set(action_list):
                    if act in level.current_action_links:
                        new_goals = new_goals + level.current_action_links[act]

                if abs(index) + 1 == len(self.graph.levels):
                    return
                elif (level, new_goals) in self.no_goods:
                    return
                else:
                    self.extract_solution(new_goals, index - 1)

        # Level-Order multiple solutions
        solution = []
        for item in self.solution:
            if item[1] == -1:
                solution.append([])
                solution[-1].append(item[0])
            else:
                solution[-1].append(item[0])

        for num, item in enumerate(solution):
            item.reverse()
            solution[num] = item

        return solution

    def goal_test(self, kb):
        return all(kb.ask(q) is not False for q in self.graph.planning_problem.goals)

    def execute(self):
        """Executes the GraphPlan algorithm for the given problem"""

        while True:
            self.graph.expand_graph()
            if (self.goal_test(self.graph.levels[-1].kb) and self.graph.non_mutex_goals(
                    self.graph.planning_problem.goals, -1)):
                solution = self.extract_solution(self.graph.planning_problem.goals, -1)
                if solution:
                    return solution

            if len(self.graph.levels) >= 2 and self.check_leveloff():
                return None


class Linearize:

    def __init__(self, planning_problem):
        self.planning_problem = planning_problem

    def filter(self, solution):
        """Filter out persistence actions from a solution"""

        new_solution = []
        for section in solution[0]:
            new_section = []
            for operation in section:
                if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                    new_section.append(operation)
            new_solution.append(new_section)
        return new_solution

    def orderlevel(self, level, planning_problem):
        """Return valid linear order of actions for a given level"""

        for permutation in itertools.permutations(level):
            temp = copy.deepcopy(planning_problem)
            count = 0
            for action in permutation:
                try:
                    temp.act(action)
                    count += 1
                except:
                    count = 0
                    temp = copy.deepcopy(planning_problem)
                    break
            if count == len(permutation):
                return list(permutation), temp
        return None

    def execute(self):
        """Finds total-order solution for a planning graph"""

        graphPlan_solution = GraphPlan(self.planning_problem).execute()
        filtered_solution = self.filter(graphPlan_solution)
        ordered_solution = []
        planning_problem = self.planning_problem
        for level in filtered_solution:
            level_solution, planning_problem = self.orderlevel(level, planning_problem)
            for element in level_solution:
                ordered_solution.append(element)

        return ordered_solution


def linearize(solution):
    """Converts a level-ordered solution into a linear solution"""

    linear_solution = []
    for section in solution[0]:
        for operation in section:
            if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                linear_solution.append(operation)

    return linear_solution

# este pedaço de código serve apenas para identificar a posição na lista da ação que pretendemos
# (a ordem das ações listadas pode ser diferente)
def applyAction(acao,a):
    """
    acao (string) a ser aplicada
    a lista de ações
    """
    if isinstance(acao, str):
        acao = expr(acao)
    for i in range(len(a)):
        if a[i].name == acao.op and a[i].args == acao.args:
            pos = i
    return a[pos]