####################################################
################# UC FUNCTIONS #####################
####################################################
import numpy as np
import math

def wuc_mov(T,a,n):
    """MoV_wUC(a) - MoV-value of alternative a of the tournament T according to the weighted Uncovered Set rule

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament
    a:  int
        an alternative
    m:  int
        number of alternatives
    n:  int
        number of matches/voters

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """
    min_wUC_value = np.inf
    Vwithouta = [x for x in T.nodes() if x!=a]
    for d in Vwithouta:
        d_mov = value_a_covered_by(T, n, a, d)
        if d_mov == -1:
            return -1
        elif d_mov < min_wUC_value:
            min_wUC_value = d_mov
    return min_wUC_value

def value_a_covered_by(T,n,a,d):
    """Size of minimum wDRS such that d covers a after reversal

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament
    n:  int
        number of matches/voters
    a:  int
        an alternative
    d:  int
        an alternative

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
        * a is wUC winner
        * d is in V\a
    """
    mov_value = 0
    d_covers_a = True
    if T[a][d]['margin']>=0:
        d_covers_a = False
        mov_value += math.ceil(n/2) - T[d][a]['weight'] +1
    Vwithoutad = [x for x in T.nodes() if x!=a and x!=d]
    for x in Vwithoutad:
        if T[a][x]['weight'] > T[d][x]['weight']:
            d_covers_a = False
            mov_value += T[a][x]['weight'] - T[d][x]['weight']
    if d_covers_a:
        return -1
    return mov_value
