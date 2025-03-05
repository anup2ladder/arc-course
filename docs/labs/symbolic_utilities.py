import re
import itertools
import math
import time
import random
from collections import Counter
from numpy.random import choice, randint
import numpy as np
from IPython.display import HTML, display, clear_output
import matplotlib.pyplot as plt
import ipywidgets as widgets
import itertools


###### Define DSLs ####

def define_bs_DSL():
    bsgrammar = {
        # A binary string can be a 0, a 1, or some transformation of string(s).
        "S": [
            (["0"], 0.2),
            (["1"], 0.2),
            # concatenate
            (["C", "(", "S", ",", "S", ")"], 0.2),
            # duplicate
            (["D", "(", "S", ")"], 0.1),
            # triplicate
            (["T", "(", "S", ")"], 0.1),
            # reverse
            (["R", "(", "S", ")"], 0.1),
            # negate
            (["N", "(", "S", ")"], 0.1)
        ]
    }
    
    BS_NONTERMINALS  = list(bsgrammar.keys())
    BS_TERMINALS = []
    for l in bsgrammar.values():
        for j in l:
            for k in j[0]:
                if k not in BS_NONTERMINALS:
                    BS_TERMINALS.append(k)
    
    BS_TERMINALS = list(set(BS_TERMINALS))
    
    bs_eval_dict = {
        "C": lambda x, y: str(x) + str(y),
        "D": lambda s: str(s) * 2,
        "T": lambda s: str(s) * 3,
        "R": lambda s: str(s)[::-1],
        "N": lambda s: str(s).translate({48: 49, 49: 48}),
        # Bit-wiseduplication: duplicate each bit in the string individually.
        "B": lambda s: "".join(c * 2 for c in str(s)),
        # Swap halves: for odd-length strings, leave the middle bit in place.
        "S": lambda s: (str(s)[(len(str(s)) // 2) + 1:] + str(s)[len(str(s)) // 2] + str(s)[:len(str(s)) // 2])
             if len(str(s)) % 2 == 1 else (str(s)[len(str(s)) // 2:] + str(s)[:len(str(s)) // 2]),
        # Rotate by one: rotate the string to the left by one position.
        "O": lambda s: str(s)[1:] + str(s)[0] if len(str(s)) > 0 else str(s),
        # Interleaving: merge two strings character by character; the remainder of the longer string is appended.
        "I": lambda x, y: "".join(a + b for a, b in zip(str(x), str(y))) +
             (str(x)[len(str(y)):] if len(str(x)) > len(str(y)) else str(y)[len(str(x)):]),
    }

    return bsgrammar, BS_NONTERMINALS, BS_TERMINALS, bs_eval_dict

def define_lt_DSL():
    ltgrammar = {
        "T": [
            (["LISTF"], 0.5),
            (["compose", "(", "LISTF", ",", "LISTF", ")"], 0.5)
        ],
        # Atomic transformations
        "LISTF": [
            (["reverse"],                  0.2),
            (["sort"],                     0.2),
            (["map_", "(", "INTF", ")"],   0.2),
            (["filter_", "(", "COND", ")"],0.2),
            (["truncate", "(", "INT", ")"],0.2)
        ],
        # Operators for map: simple arithmetic operations on x.
        "INTF": [
            (["plus", "(", "INT", ")"],    0.25),
            (["minus", "(", "INT", ")"],   0.25),
            (["times", "(", "INT", ")"],   0.25),
        ],
        # Conditions for filter: e.g., keeping even or odd numbers.
        "COND": [
            (["even"], 0.5),
            (["gt", "(", "INT", ")"],  0.3),
            (["not_", "(", "COND", ")"], 0.1),
            (["and_", "(", "COND", ",", "COND", ")"], 0.1),
            (["or_", "(", "COND", ",", "COND", ")"], 0.1)
        ],
        "INT": [
            ([str(i)], 1/5) for i in range(1,6)
        ]
    }
    
    LT_NONTERMINALS  = list(ltgrammar.keys())
    LT_TERMINALS = []
    for l in ltgrammar.values():
        for j in l:
            for k in j[0]:
                if k not in LT_NONTERMINALS:
                    LT_TERMINALS.append(k)
    
    LT_TERMINALS = list(set(LT_TERMINALS))
    
    lt_eval_dict = {
        # Composes two list transformation functions.
        "compose":  lambda f, g: lambda L: g(f(L)),
        # Basic list transformations.
        "reverse":  lambda L: list(reversed(L)),
        "sort":     lambda L: sorted(L),
        "truncate": lambda i: lambda L: L[:i],
        # Higher-order functions that expect a function and return a list transformation.
        "map_":     lambda f: lambda L: [f(x) for x in L],
        "filter_":  lambda f: lambda L: [x for x in L if f(x)],
        # Integer transformation functions used inside map.
        "plus":     lambda n: lambda x: x + n,
        "minus":    lambda n: lambda x: x - n,
        "times":    lambda n: lambda x: x * n,
        # Predicates for filtering.
        "even":     lambda x: x % 2 == 0,
        "gt":       lambda i: lambda x: x>i,
        "and_":     lambda f, g: lambda x: f(x) and g(x),
        "or_":      lambda f, g: lambda x: f(x) or g(x),
        "not_":     lambda f: lambda x: not f(x),
    }

    return ltgrammar, LT_NONTERMINALS, LT_TERMINALS, lt_eval_dict

ltgrammar, lt_nonterminals, lt_terminals, lt_eval_dict = define_lt_DSL()
bsgrammar, bs_nonterminals, bs_terminals, bs_eval_dict = define_bs_DSL()

###### For Lab 1 ######
import cairosvg
import io
from PIL import Image, ImageFilter
from scipy.signal import sawtooth

def normalize(arr):
    # Transform the array so that it sums to 1
    return arr / np.sum(arr)


def apply_rule(match, grammar):
    """
    Accepts both PCFGs (where the keys of grammar
    are lists of tuples (unnormalized prob, right hand side)
    and CFGs (where the keys are lists of right hand side strings.
    """
    match_txt = grammar[match.group(0)]
    try:
        # if PCFG
        probs, subs = zip(*match_txt)
    except ValueError:
        # if CFG
        subs = match_txt
        probs = [1]*len(subs)
    sub = choice(subs, p=normalize(probs))
    return sub


def complete(sentence, grammar):
    # if there are still '<' that means there are still nonterminals
    while '<' in sentence:
        # substitute the first nonterminal applying 
        # one of the rules in the grammar
        sentence = re.sub(
            # Greedy search of an expression between
            # angle brackets.
            '<(.*?)>', 
            lambda match: apply_rule(match, grammar), 
            sentence
        )
    return sentence


def get_svg(screen=None):
        """Shows the SVG code for the image to the screen."""
        if screen is None:
            screen = t.Screen()
        header = ("""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">\n""").format(
            w=screen.window_size[0],
            h=screen.window_size[1]) 
        header += ("""<rect width="100%" height="100%" style="fill:{fillcolor};stroke:{kolor};stroke-width:1" />\n""").format(
            fillcolor=screen.background_color,
            kolor=screen.border_color)
        # lines = screen._svg_drawlines_string.replace("/>","/>\n")
        image = screen._generateSvgLines().replace("/>","/>\n") 
        output = header + image + "</svg>"
        return output

def blur_img(img, p_noise=0.):
    # whether to switch pixel or not
    # noise should be maximal when p_noise is 1
    p_noise = p_noise/2
    salt_and_pepper = np.where(
        np.random.choice(
            [0,1],
            size=img.shape,
            p=[1-p_noise, p_noise]
        ),
        img,
        1-img
    )
    return salt_and_pepper

def svg_to_img(svg, p_noise=0):
    """
    Load an SVG file and return image in Numpy array.
    Modified version of function from:
    https://stackoverflow.com/questions/55440483/
    how-can-i-optimize-the-palette-image-size-with-pil
    /55442505#55442505
    """
    # Make memory buffer
    mem = io.BytesIO()
    # Convert SVG to PNG in memory
    cairosvg.svg2png(
        bytestring=svg, 
        write_to=mem,
    )
    # Convert PNG to Pillow object
    img = np.array(Image.open(mem))
    # Remove greys and normalize in [0,1]
    # and take only first value of 3 RGB 
    # (They're all the same since original image
    # is in shades of grey)
    img = np.round(img[:,:,0]/img.max())
    return blur_img(
        np.array(img), 
        p_noise
    )


###### For lab 2 ######

def interpret(string, eval_dict):
    # in case I decide to do something fancy later
    return eval(string, eval_dict)


###### Visualization utils

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def compute_global_limits_mh(trace):
    xs, ys = [], []
    for rec in trace:
        if rec['current_coord'] is not None:
            x, y = rec['current_coord']
            xs.append(x)
            ys.append(y)
        if rec['proposal_coord'] is not None:
            x, y = rec['proposal_coord']
            xs.append(x)
            ys.append(y)
    # Use a margin factor
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = 0.1
    return ([xmin * (1 - margin), xmax * (1 + margin)],
            [ymin * (1 - margin), ymax * (1 + margin)])

def plot_mh_trace_upto(trace, upto, global_xlims, global_ylims):
    """
    Plot the MH chain trace from iteration 0 up to iteration 'upto'.
    Accepted moves are shown as solid green arrows; rejected proposals as red dashed arrows.
    Accepted state points are shown as blue circles annotated with their formula.
    The plot uses log scale (fixed axes based on the global limits).
    """
    accepted_coords = []
    proposals = []  # Each element: (x_old, y_old, x_prop, y_prop, accepted)
    
    # The initial accepted state (iteration 0).
    prev_coord = trace[0]['current_coord']
    accepted_coords.append(prev_coord)
    
    for rec in trace[1:upto+1]:
        if rec['proposal_coord'] is None:
            continue
        x_old, y_old = prev_coord
        x_prop, y_prop = rec['proposal_coord']
        accepted_move = rec['accepted']
        proposals.append((x_old, y_old, x_prop, y_prop, accepted_move))
        if accepted_move:
            prev_coord = rec['current_coord']
            accepted_coords.append(prev_coord)
    
    # Start plotting.
    plt.figure(figsize=(10, 6))
    
    # Plot accepted chain points.
    accepted_x = [pt[0] for pt in accepted_coords]
    accepted_y = [pt[1] for pt in accepted_coords]
    plt.plot(
        accepted_x, 
        accepted_y, 
        marker='o', 
        color='blue', 
        label="Accepted Chain", 
        zorder=3
    )

    # Annotate only the last two accepted points with their formulas.
    accepted_iter = [rec['iteration'] for rec in trace[:upto+1] if rec['accepted'] is True or rec['iteration'] == 0]
    accepted_expr = [rec['expression'] for rec in trace[:upto+1] if rec['accepted'] is True or rec['iteration'] == 0]
    # Only take the last two accepted points (if there are at least two).
    if len(accepted_expr) >= 2:
        for x, y, expr in zip(accepted_x[-2:], accepted_y[-2:], accepted_expr[-2:]):
            plt.text(x, y, expr, fontsize=8, color='darkblue', ha='right', va='top', rotation=-270)
    else:
        for x, y, expr in zip(accepted_x, accepted_y, accepted_expr):
            plt.text(x, y, expr, fontsize=8, color='darkblue', ha='right', va='top', rotation=-270)
    
    # Annotate accepted points with the formula.
    # We'll match the accepted points to the corresponding iterations in the trace.
    # accepted_iter = [rec['iteration'] for rec in trace if rec['accepted'] is True or rec['iteration'] == 0]
    # accepted_expr = [rec['expression'] for rec in trace if rec['accepted'] is True or rec['iteration'] == 0]
    # for x, y, expr in zip(accepted_x, accepted_y, accepted_expr):
    #     plt.text(x, y, expr, fontsize=8, color='darkblue', ha='right', va='bottom')
    
    # Draw arrows for each proposal.
    for (x_old, y_old, x_prop, y_prop, accepted_move) in proposals:
        if accepted_move:
            plt.plot([x_old, x_prop], [y_old, y_prop], c='green', linestyle='--')
        else:
            plt.plot([x_old, x_prop], [y_old, y_prop], c='red', linestyle='--')
    
    plt.xlabel("Prior (log scale)")
    plt.ylabel("Likelihood (log scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"MH Chain Trace up to Iteration {upto}")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.xlim(*global_xlims)
    plt.ylim(*global_ylims)
    plt.ylim(10e-20, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_global_limits_smc(trace, grammar, data, eval_dict, likelihoodf):
    xs, ys = [], []
    for state in trace:
        for p in state['particles']:
            x = compute_tree_probability(p['tree'], grammar)
            expr = "".join(tree_to_sentence(p['tree']))
            transf = interpret(expr, eval_dict)
            y = likelihoodf(transf, data)
            xs.append(x)
            ys.append(y)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = 0.1
    return ([xmin * (1 - margin), xmax * (1 + margin)],
            [ymin * (1 - margin), ymax * (1 + margin)])


def plot_state_2d(state, grammar, data, global_xlims, global_ylims, eval_dict, likelihoodf):
    """
    For a given state, plot a 2D scatter plot (log–log scale) where:
      - x-axis is the prior,
      - y-axis is the likelihood,
      - the point color reflects the particle weight,
      - point size is increased,
      - if the state is post-mutation, red arrows show the movement.
    """
    priors, likelihoods, weights = [], [], []
    labels = []
    for p in state['particles']:
        prior_val = compute_tree_probability(p['tree'], grammar)
        try:
            expr = "".join(tree_to_sentence(p['tree']))
            transf = interpret(expr, eval_dict)
            like_val = likelihoodf(transf, data)
        except Exception:
            like_val = 0.0
        priors.append(prior_val)
        likelihoods.append(like_val)
        weights.append(p['weight'])
        labels.append(expr)
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(priors, likelihoods, c=weights, s=100, cmap='viridis', alpha=0.8)
    plt.colorbar(sc, label="Weight")
    plt.xlabel("Prior")
    plt.ylabel("Likelihood")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(*global_xlims)
    plt.ylim(*global_ylims)
    
    # If in the mutation stage, draw arrows (using lines) from old to new.
    if state['stage'] == 'post-mutation':
        for p in state['particles']:
            if 'old_coord' in p and 'new_coord' in p:
                x_old, y_old = p['old_coord']
                x_new, y_new = p['new_coord']
                plt.plot([x_old, x_new], [y_old, y_new], c='black', linestyle='--', alpha=0.7)
    
    plt.title(f"Iteration {state['iteration']} - Stage: {state['stage']}")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


##### Tree generation and sampling utils

def generate_tree(symbol, grammar):
    """
    Recursively generates a parse tree starting from a given symbol.
    A tree is represented as a tuple: (symbol, children)
    For terminals (symbols not in grammar) the children list is empty.
    """
    # terminal symbol
    if symbol not in grammar:  
        return (symbol, [])
    
    productions = grammar[symbol]
    # Sample a production according to its probability
    r = random.random()
    cumulative = 0.0
    for production, prob in productions:
        cumulative += prob
        if r < cumulative:
            # For each symbol in the production, generate its subtree.
            children = [generate_tree(sym, grammar) for sym in production]
            return (symbol, children)
    # Fallback (should not occur if probabilities sum to 1)
    production, _ = productions[-1]
    children = [generate_tree(sym, grammar) for sym in production]
    return (symbol, children)


def tree_to_sentence(tree):
    """
    Converts a parse tree into a sentence (a list of terminal tokens).
    """
    symbol, children = tree
    if not children:
        return [symbol]
    words = []
    for child in children:
        words.extend(tree_to_sentence(child))
    return words


def get_nonterminal_nodes(tree, index=()):
    """
    Recursively collects all nodes that are non-terminals (i.e. have children)
    in the parse tree along with their index path.
    
    Each entry is a tuple (node, path) where 'path' is a tuple of indices indicating 
    how to locate the node within the tree.
    """
    symbol, children = tree
    nodes = []
    if children:  # nonterminal (or at least an internal node)
        nodes.append((tree, index))
        for i, child in enumerate(children):
            nodes.extend(get_nonterminal_nodes(child, index + (i,)))
    return nodes


def set_subtree(tree, path, new_subtree):
    """
    Replaces the subtree in 'tree' at the given 'path' with 'new_subtree' and 
    returns the new tree.
    """
    if not path:
        return new_subtree
    index = path[0]
    symbol, children = tree
    new_children = children.copy()
    new_children[index] = set_subtree(children[index], path[1:], new_subtree)
    return (symbol, new_children)


def mutate_tree(tree, grammar):
    """
    Mutation operator: Selects a random nonterminal node in the parse tree and 
    regenerates its subtree using the PCFG.
    """
    nodes = get_nonterminal_nodes(tree)
    if not nodes:
        return tree  # No nonterminal available to mutate.
    # Randomly select a nonterminal node
    node, path = random.choice(nodes)
    symbol, _ = node
    # Regenerate the subtree starting from the selected nonterminal symbol.
    new_subtree = generate_tree(symbol, grammar)
    # Replace the old subtree with the new one.
    mutated_tree = set_subtree(tree, path, new_subtree)
    return mutated_tree


##### Unnormalized posterior utils

def compute_tree_probability(tree, grammar):
    """
    Recursively computes the probability of a parse tree under the PCFG.
    
    The tree is assumed to be a tuple (symbol, children). For a terminal symbol (i.e. not in grammar),
    the probability is 1. For a nonterminal, we try to match one of the productions (the right-hand side)
    to the children of the tree. The overall probability is the product of the production probability 
    and the probabilities of the subtrees.
    
    Raises a ValueError if no production in the grammar can explain the given children.
    """
    symbol, children = tree

    # Terminal: probability 1.
    if symbol not in grammar:
        return 1.0

    # For nonterminals, look for a production that matches the structure of 'children'.
    for production, prod_prob in grammar[symbol]:
        # The production is a list of symbols. It must have the same number of children.
        if len(production) != len(children):
            continue
        
        # Check that each symbol in the production corresponds to the child's root symbol.
        match = True
        for prod_sym, child in zip(production, children):
            child_symbol = child[0]
            if prod_sym != child_symbol:
                match = False
                break
        
        # If a matching production is found, multiply its probability with the probabilities of the children.
        if match:
            child_prob = 1.0
            for child in children:
                child_prob *= compute_tree_probability(child, grammar)
            return prod_prob * child_prob

    # If no production matches the structure of the children, raise an error.
    raise ValueError("No matching production found for tree node: {}".format(tree))


def compute_likelihood_lt(transformation, data, match_prob=0.99, length_mismatch_prob=1e-6):
    """
    Computes the likelihood of the data given a transformation function generated by the PCFG.

    Parameters:
        transformation: A function (generated via eval on the PCFG output) that takes a list of ints 
                        and returns a transformed list of ints.
        data: A list of tuples (input, observed_output), where each is a list of integers.
        match_prob: Probability that a predicted number equals the observed number (i.e. no error).
        mismatch_prob: Probability that a predicted number does not match the observed number.
        length_mismatch_prob: Likelihood assigned to an input/output pair if the predicted and observed 
                              lists differ in length.
    
    Returns:
        The overall likelihood (a float) computed as the product over all input/output pairs.
    """
    total_likelihood = 1.0

    for inp, observed in data:
        predicted = transformation(inp)
        
        # If the lengths don't match, assign a fixed small probability.
        if len(predicted) != len(observed):
            total_likelihood *= length_mismatch_prob
        else:
            # Compare each element.
            for p, o in zip(predicted, observed):
                if p == o:
                    total_likelihood *= match_prob
                else:
                    total_likelihood *= 1-match_prob

    return total_likelihood

def compute_likelihood_bs(predicted_string, data_string, match_prob=0.99, length_mismatch_prob=1e-6):
    # If the lengths don't match, assign a fixed small probability.
    # The grammar produces an int iff the string is just 0 or 1
    if len(str(predicted_string)) != len(data_string):
        return length_mismatch_prob
    
    total_likelihood = 1.0
    for p, o in zip(str(predicted_string), str(data_string)):
        if p == o:
            total_likelihood *= match_prob
        else:
            total_likelihood *= 1-match_prob
    return total_likelihood


def compute_unnormalized_posterior(tree, grammar, data, eval_dict, likelihoodf, lik_params=None, prior_params=None):
    """
    Computes an unnormalized posterior for a given sentence.
    """
    if lik_params is None:
        lik_params = dict()
    if prior_params is None:
        prior_params = dict()
    prior = compute_tree_probability(tree, grammar, **prior_params)
    t = ''.join(tree_to_sentence(tree))
    transf = interpret(t, eval_dict)
    likelihood = likelihoodf(transf, data, **lik_params)
    return prior*likelihood


#### Enumeration

def is_complete(tree, grammar):
    """A parse tree (symbol, children) is complete if every leaf is terminal (not in grammar)."""
    symbol, children = tree
    if not children:
        return symbol not in grammar
    return all(is_complete(child, grammar) for child in children)


def enumerate_trees(symbol, grammar, max_depth=4, current_depth=0):
    """
    Lazily enumerates only those parse trees (as tuples) that are complete,
    i.e. every leaf is terminal (a symbol not in grammar).
    
    Parameters:
      symbol: The current symbol to expand.
      grammar: A dictionary mapping nonterminals to a list of productions.
               Each production is a tuple (production_list, probability).
      max_depth: Maximum recursion depth.
      current_depth: Current recursion depth.
      
    Yields:
      Only complete parse trees.
    """
    # If symbol is terminal, yield a leaf.
    if symbol not in grammar:
        yield (symbol, [])
        return
    
    # If we've reached max depth and symbol is still nonterminal, yield nothing.
    if current_depth >= max_depth:
        return

    # For each production for the symbol...
    for production, _ in grammar[symbol]:
        # For each symbol in the production, recursively enumerate its complete trees.
        child_options = []
        skip_prod = False
        for sym in production:
            # Get all complete trees for sym at the next depth.
            trees = list(enumerate_trees(sym, grammar, max_depth, current_depth + 1))
            if not trees:
                # This production cannot yield a complete tree.
                skip_prod = True
                break
            child_options.append(trees)
        if skip_prod:
            continue
        # Cartesian product: one tree per combination of children.
        for children_tuple in itertools.product(*child_options):
            tree = (symbol, list(children_tuple))
            if is_complete(tree, grammar):
                yield tree


def enumerate_full_sentences(symbol, grammar, max_depth=4, current_depth=0):
    gen = enumerate_trees(
        symbol, 
        grammar, 
        max_depth, 
        current_depth
    )
    for tree in gen:
        sentence = "".join(tree_to_sentence(tree)) 
        if all([y not in sentence for y in grammar.keys()]):
            yield sentence


def enumerate_trees_bottomup(start, grammar, max_level=4, eval_env=None, are_same=None):
    """
    Enumerates complete parse trees using a bottom-up (dynamic programming) approach.
    
    A parse tree is represented as a tuple: (symbol, children)
    where children is a tuple of child trees.
    
    Optionally, if both eval_env (a dictionary for eval) and are_same (a function to compare meanings)
    are provided, then on each level semantic pruning is performed: for each nonterminal, if a tree
    with a given meaning has already been found at a lower level, any new tree with an equivalent
    meaning is discarded.
    
    Parameters:
      start: The start symbol.
      grammar: A dict mapping nonterminals to a list of productions.
               Each production is a tuple (production_list, probability).
      max_level: Maximum number of production steps.
      eval_env: A dictionary to pass as globals to eval (optional).
      are_same: A function taking two meaning values and returning True if they are equivalent (optional).
      
    Returns:
      A list of complete parse trees for the start symbol.
    """
    # dp maps each nonterminal to a list of (tree, meaning) pairs.
    # (If semantic pruning is inactive, meaning will be None.)
    dp = { nt: [] for nt in grammar }
    
    # For a terminal symbol, the only complete tree is (sym, ()).
    def terminal_tree(sym):
        tree = (sym, ())
        if eval_env is not None and are_same is not None:
            sentence = "".join(tree_to_sentence(tree))
            try:
                meaning = eval(sentence, eval_env)
            except Exception:
                meaning = None
        else:
            meaning = None
        return [(tree, meaning)]
    
    # Build trees level by level.
    for level in range(1, max_level + 1):
        new_dp = { nt: [] for nt in grammar }
        for A in grammar:
            for production, _ in grammar[A]:
                child_options = []
                valid = True
                for sym in production:
                    if sym in grammar:
                        if not dp[sym]:
                            valid = False
                            break
                        else:
                            child_options.append(dp[sym])
                    else:
                        child_options.append(terminal_tree(sym))
                if not valid:
                    continue
                # For each combination of complete subtrees...
                for children_tuple in itertools.product(*child_options):
                    # Each element in children_tuple is a (tree, meaning) pair.
                    child_trees = tuple(item[0] for item in children_tuple)
                    tree = (A, child_trees)
                    if eval_env is not None and are_same is not None:
                        sentence = "".join(tree_to_sentence(tree))
                        try:
                            meaning = eval(sentence, eval_env)
                        except Exception:
                            meaning = None
                    else:
                        meaning = None
                    # Semantic pruning: if a tree with an equivalent meaning is already known for A, skip.
                    skip = False
                    if eval_env is not None and are_same is not None and meaning is not None:
                        for (_, existing_meaning) in dp[A]:
                            if are_same(existing_meaning, meaning):
                                skip = True
                                break
                        if not skip:
                            for (_, existing_meaning) in new_dp[A]:
                                if are_same(existing_meaning, meaning):
                                    skip = True
                                    break
                    if not skip:
                        new_dp[A].append((tree, meaning))
        any_new = False
        for A in grammar:
            old_size = len(dp[A])
            dp[A].extend(new_dp[A])
            if len(dp[A]) > old_size:
                any_new = True
        if not any_new:
            break
    return [tree for (tree, meaning) in dp.get(start, [])]

def enumerate_full_sentences_bottomup(start, grammar, max_level=4, eval_env=None, are_same=None):
    """
    Converts each complete parse tree from the bottom-up enumerator to a sentence.
    Assumes you have a function tree_to_sentence(tree) that returns a list of tokens.
    Yields only sentences that contain no nonterminals.
    """
    for tree in enumerate_trees_bottomup(start, grammar, max_level, eval_env, are_same):
        sentence = "".join(tree_to_sentence(tree))
        if all(sym not in grammar for sym in sentence):
            yield sentence


##### MH sampling

def propose_tree(current_tree, grammar):
    """
    Propose a new tree by selecting a random nonterminal in current_tree,
    regenerating that subtree, and replacing it.
    Returns:
      new_tree: The proposed tree.
      proposal_ratio: The factor q(current|proposal)/q(proposal|current) computed as (N(current)/N(new)) * (P(old)/P(new))
      old_subtree: The subtree that was replaced.
      new_subtree: The new subtree.
    """
    nodes = get_nonterminal_nodes(current_tree)
    if not nodes:
        # No proposal possible; return current tree and ratio 1.
        return current_tree, 1.0, None, None
    node, path = random.choice(nodes)
    symbol, _ = node
    old_subtree = node
    new_subtree = generate_tree(symbol, grammar)
    new_tree = set_subtree(current_tree, path, new_subtree)
    N_current = len(get_nonterminal_nodes(current_tree))
    N_new = len(get_nonterminal_nodes(new_tree))
    p_old = compute_tree_probability(old_subtree, grammar)
    p_new = compute_tree_probability(new_subtree, grammar)
    proposal_ratio = (N_current / N_new) * (p_old / p_new) if p_new > 0 else 0
    return new_tree, proposal_ratio, old_subtree, new_subtree


def get_coordinates(tree, grammar, data, eval_dict, likelihoodf):
    """Return (prior, likelihood) coordinates for a tree."""
    prior = compute_tree_probability(tree, grammar)
    try:
        expr = "".join(tree_to_sentence(tree))
        transf = interpret(expr, eval_dict)
        like = likelihoodf(transf, data)
    except Exception:
        like = 0.0
    return (prior, like)


def mh_sampler(grammar, data, starting, eval_dict, likelihoodf, num_iterations=100):
    """
    Runs a Metropolis–Hastings sampler and records the trace.
    Returns a list of records, one per iteration, where each record is a dict:
      - 'iteration': iteration number
      - 'current_tree': the accepted tree at the end of the iteration
      - 'current_coord': (prior, likelihood) of the accepted tree
      - 'proposal_coord': (prior, likelihood) of the proposed tree
      - 'accepted': True if proposal was accepted, False otherwise
      - 'alpha': acceptance probability
    The initial state (iteration 0) is recorded with no proposal.
    """
    trace = []
    current_tree = generate_tree(starting, grammar)
    current_post = compute_unnormalized_posterior(current_tree, grammar, data, eval_dict, likelihoodf)
    current_coord = get_coordinates(current_tree, grammar, data, eval_dict, likelihoodf)
    trace.append({
        'iteration': 0,
        'current_tree': current_tree,
        'current_coord': current_coord,
        'proposal_coord': None,
        'accepted': None,
        'alpha': None,
        'expression': "".join(tree_to_sentence(current_tree))
    })

    out = display(progress(0, 100), display_id=True)
    for it in range(1, num_iterations+1):
        proposal_tree, proposal_ratio, old_subtree, new_subtree = propose_tree(current_tree, grammar)
        new_post = compute_unnormalized_posterior(proposal_tree, grammar, data, eval_dict, likelihoodf)
        alpha = 1.0
        if current_post > 0:
            alpha = min(1, (new_post / current_post) * proposal_ratio)
        else:
            alpha = 1.0
        r = random.random()
        accepted = (r < alpha)
        proposal_coord = get_coordinates(proposal_tree, grammar, data, eval_dict, likelihoodf)
        if accepted:
            current_tree = proposal_tree
            current_post = new_post
            current_coord = proposal_coord
        # Record the iteration
        trace.append({
            'iteration': it,
            'current_tree': current_tree,
            'current_coord': current_coord,
            'proposal_coord': proposal_coord,
            'accepted': accepted,
            'alpha': alpha,
            'expression': "".join(tree_to_sentence(current_tree))
        })
        out.update(progress(it, num_iterations))
    return trace


###### SMC sampling
def smc_sampler(grammar, data, starting, eval_dict, 
                num_particles=100, num_iterations=10, resample_prop=0.5, 
                likelihoodf=compute_likelihood_lt):
    """
    Runs an SMC sampler where:
      - Particles are parse trees (with associated sentences) generated from a PCFG.
      - Utility (weight) is computed via an unnormalized posterior that includes both 
        the PCFG probability and the likelihood on the data.
      - Mutation is performed via subtree regeneration.
      - Only a set proportion (resample_prop) of the particles are resampled at each iteration.
    Also records intermediate states.
    
    Each state is a dictionary with:
      - 'iteration': iteration number
      - 'stage': one of 'initial', 'pre-resampling', 'post-resampling', 'post-mutation'
      - 'particles': list of particle dictionaries (each with keys 'tree', 'sentence', 'weight')

    Parameters:
      grammar: The PCFG grammar.
      data: List of (input, observed_output) pairs.
      starting: The start symbol for tree generation.
      num_particles: Total number of particles.
      num_iterations: Number of iterations to run.
      resample_prop: Proportion of particles to resample (0.0 < resample_prop <= 1.0).
    
    Returns:
      The final list of particles.
    """
    states = []
    particles = []
    # Initialize particles.
    for _ in range(num_particles):
        tree = generate_tree(starting, grammar)
        sentence = tree_to_sentence(tree)
        weight = compute_unnormalized_posterior(tree, grammar, data, eval_dict, likelihoodf)
        particles.append({'tree': tree, 'sentence': sentence, 'weight': weight})
    states.append({'iteration': 0, 'stage': 'initial', 'particles': [p.copy() for p in particles]})

    out = display(progress(0, 100), display_id=True)
    for iteration in range(1, num_iterations+1):
        # Stage 1: Pre-resampling (compute weights)
        weights = [compute_unnormalized_posterior(p['tree'], grammar, data, eval_dict, likelihoodf) for p in particles]
        for i, p in enumerate(particles):
            p['weight'] = weights[i]
        states.append({'iteration': iteration, 'stage': 'pre-resampling', 'particles': [p.copy() for p in particles]})
        
        total_weight = sum(weights)
        if total_weight == 0:
            normalized_weights = [1.0/num_particles]*num_particles
        else:
            normalized_weights = [w/total_weight for w in weights]
        
        # Stage 2: Resampling a proportion of particles.
        if resample_prop < 1.0:
            num_to_resample = int(num_particles * resample_prop)
            resample_indices = random.sample(range(num_particles), num_to_resample)
            for idx in resample_indices:
                chosen_idx = random.choices(range(num_particles), weights=normalized_weights)[0]
                particles[idx] = particles[chosen_idx].copy()
        else:
            new_particles = []
            for _ in range(num_particles):
                chosen_idx = random.choices(range(num_particles), weights=normalized_weights)[0]
                new_particles.append(particles[chosen_idx].copy())
            particles = new_particles
        states.append({'iteration': iteration, 'stage': 'post-resampling', 'particles': [p.copy() for p in particles]})
        
        # Stage 3: Mutation
        # Record pre-mutation coordinates for each particle.
        pre_mutation_coords = []
        for p in particles:
            prior_val = compute_tree_probability(p['tree'], grammar)
            try:
                expr = "".join(tree_to_sentence(p['tree']))
                transf = interpret(expr, eval_dict)
                like_val = likelihoodf(transf, data)
            except Exception:
                like_val = 0.0
            pre_mutation_coords.append( (prior_val, like_val) )
        # Now mutate and record new coordinates.
        for i, p in enumerate(particles):
            mutated_tree = mutate_tree(p['tree'], grammar)
            p['tree'] = mutated_tree
            p['sentence'] = tree_to_sentence(mutated_tree)
            p['weight'] = compute_unnormalized_posterior(mutated_tree, grammar, data, eval_dict, likelihoodf)
            # Compute new coordinates.
            new_prior = compute_tree_probability(mutated_tree, grammar)
            try:
                new_expr = "".join(tree_to_sentence(mutated_tree))
                transf = interpret(new_expr, eval_dict)
                new_like = likelihoodf(transf, data)
            except Exception:
                new_like = 0.0
            p['old_coord'] = pre_mutation_coords[i]
            p['new_coord'] = (new_prior, new_like)
        states.append({'iteration': iteration, 'stage': 'post-mutation', 'particles': [p.copy() for p in particles]})
        out.update(progress(iteration, num_iterations))
    
    return states