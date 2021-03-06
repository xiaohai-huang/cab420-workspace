"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6 AIMA)."""

from collections import defaultdict

from  search import Problem, depth_first_tree_search, depth_first_graph_search

class CSP(Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        csp_vars    A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases. (For example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
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

    >>> depth_first_graph_search(australia)
    <Node (('WA', 'B'), ('Q', 'B'), ('T', 'B'), ('V', 'B'), ('SA', 'G'), ('NT', 'R'), ('NSW', 'R'))>
    """

    def __init__(self, csp_vars, domains, neighbors, constraints):
        "Construct a CSP problem. If csp_vars is empty, it becomes domains.keys()."
        csp_vars = csp_vars or domains.keys()
        self.csp_vars=csp_vars
        self.domains=domains
        self.neighbors=neighbors 
        self.constraints=constraints
        self.initial=()
        self.curr_domains=None
        self.nassigns=0

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
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])
        nc = 0
        for v2 in self.neighbors[var]:
            if conflict(v2):
                nc += 1
        return nc

    def display(self, assignment):
        "Show a human-readable representation of the CSP."
        # Subclasses can print in a prettier way, or display with a GUI
        print ('CSP:', self, 'with assignment:', assignment)

    ## These methods are for the tree- and graph-search interface:

    def actions(self, state):
        """
           Return the list of applicable nonconflicting
           assignments to the first unassigned variable.
         """
        if len(state) == len(self.csp_vars):
            return []
        else:
            assignment = dict(state)
            for v in self.csp_vars:
                if v not in assignment:
                    var = v
                    break
            else:
                return []
            # var is an unassigned variable (wrt to assignment)
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, a): # a is a pair (var, val)
        "Perform an action and return the new state."
        var, val = a
        return state + ((var, val),)

    def goal_test(self, state):
        "The goal is to assign all csp_vars, with all constraints satisfied."
        assignment = dict(state)
        return (len(assignment) == len(self.csp_vars) and
                all(self.nconflicts(var, assignment[var], assignment) == 0 for var in self.csp_vars))


    ## This is for min_conflicts search

    def conflicted_vars(self, current):
        "Return a list of variables in current assignment that are in conflict"
        return [var for var in self.csp_vars
                if self.nconflicts(var, current[var], current) > 0]



#______________________________________________________________________________
# Min-conflicts hillclimbing search for CSPs

import random
def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0])
    n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
            n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                best = x
    return best

def min_conflicts(csp, max_steps=100000):
    """
    Solve a CSP by stochastic hillclimbing on the number of conflicts.
    Return None is no solution found,
        otherwise return  a dictionary of proper assignments
    """
    # Generate a complete assignment for all csp_vars (probably with conflicts)
    csp.current = current = {}
    for var in csp.csp_vars:
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

def parse_neighbors(neighbors, csp_vars=[]):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors.  The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name.  If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z')
    {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    """
    n_dict = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        n_dict.setdefault(A, [])
        for B in Aneighbors.split():
            n_dict[A].append(B)
            n_dict[B].append(A)
    return n_dict

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
#

if __name__ == "__main__":
    import time
    
#    country  = australia # one of the map colouring CSP's
#    country  = france # one of the map colouring CSP's
    country  = usa # one of the map colouring CSP's

    t0 = time.time()
#    sol = depth_first_graph_search(country)
    sol = depth_first_tree_search(country)
    t1 = time.time()
    print ("DFS solution -> ", sol)
    print ("DFS Solver took ",t1-t0, ' seconds')

    
    t0 = time.time()
    sol = min_conflicts(country)
    t1 = time.time()
    print ("Min-conflict solver found -> ", sol)
    print ("Min-conflict Solver took ",t1-t0, ' seconds')
        
##    print min_conflicts(france)

##    {'WA': 'B', 'Q': 'B', 'T': 'G', 'V': 'B', 'SA': 'R', 'NT': 'G', 'NSW': 'G'}
