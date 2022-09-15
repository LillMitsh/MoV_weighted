####################################################
################# SC FUNCTIONS #####################
####################################################
import networkx as nx
import numpy as np

from splitcycle_master.voting.voting_methods import split_cycle_faster_mg

def sc_mov(T, mgg, a, n):
    """MoV_SC(a) - MoV-value of alternative a of the tournament T according to the SplitCycle rule

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
    sc_winner = split_cycle_faster_mg(mgg)
    if not a in sc_winner:
        return -1
    vwithouta = [x for x in T.nodes() if x != a]
    min_value = np.inf
    for l in range(n % 2, n, 2):
        if l == 0:
            continue
        for d in vwithouta:
            l_d_value = sc_mov_for(T, a, l, d)
            if l_d_value < min_value:
                min_value = l_d_value
    return min_value

def sc_mov_for(T, a, l, d):
    """Size of minimum wDRS such that d dominates a with margin l in the margin graph after reversal

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament
    a:  int
        an alternative
    l:  int
        margin value
    b:  int
        an alternative

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
        * a is Split Cycle winner
        * b is in V\a
        * l is in [n%2 , n; 2]\0
    """
    G = nx.DiGraph()
    G.add_nodes_from(T.nodes())
    for (u,v) in T.edges():
        if (u,v) != (a,d) and (u,v) != (d,a):
            if T[u][v]['margin'] >= l:
                G.add_edge(u,v, capacity = T[u][v]['margin'] - (l-2))
    max_flow = nx.maximum_flow(G,a,d)
    return 0.5*(max_flow[0] + l - T[d][a]['margin'])
