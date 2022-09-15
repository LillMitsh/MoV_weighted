import numpy as np
import math
####################################################
################# BO FUNCTIONS #####################
####################################################

def bo_scores(T):
    """The Borda scores for all alternatives in tournament T

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """
    scores = [None]*len(T.nodes)
    for v in T.nodes():
        s_BO = 0
        for (v,u) in T.out_edges(v):
            s_BO += T[v][u]['weight']
        scores[v]=s_BO
    return scores

def bo_winners(T):
    """The Borda winners of tournament T

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """
    max_degree = max(bo_scores(T))
    bo_winners = [v for v in T.nodes() if bo_scores(T)[v] == max_degree]
    return bo_winners

def bo_runner_ups(T):
    """The Runner-ups of tournament T

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """
    winners = bo_winners(T)
    if not set(winners) == set(T.nodes()):
        scores = bo_scores(T)
        scores.remove(max(scores))
        runner_degree = max(scores)
        runner_ups = [v for v in T.nodes() if scores[v] == runner_degree]
        return runner_ups
    else:
        return []

def bo_mov(T, a):
    """MoV_BO(a) - MoV-value of alternative a of the tournament T according to Borda's rule

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament
    a:  int
        an alternative

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """
    winners = bo_winners(T)
    scores = bo_scores(T)
    if not (a in winners):
        return -1
    if len(winners) > 1:
        return 1
    else:
        min_bo_value = np.inf
        Vwithouta_b = [x for x in T.nodes() if x != a]
        for b in Vwithouta_b:
            b_mov = bo_DRS_by(T, a, b, scores)
            if b_mov < min_bo_value:
                min_bo_value = b_mov
        return min_bo_value


def bo_DRS_by(T, a, b, scores):
    """Size of minimum wDRS such that b is in the winning set instead of a after reversal

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament
    a:  int
        an alternative
    b:  int
        an alternative
    scores:     1d numpy array
        list of Borda scores for each alternative

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
        * a is Borda winner
        * b is in V\a
    """
    sbo_b = scores[b]  # supposed to take over a
    sbo_a = scores[a]
    if T[a][b]['weight'] >= math.floor((sbo_a - sbo_b) / 2) + 1:  # max reverse possible >= needed reverse to put b above a
        return math.floor((sbo_a - sbo_b) / 2) + 1  # done
    else:
        bo_DRS_by_b__value = T[a][b]['weight']  # reverse as much as possible (and more as needed)
        sbo_a = sbo_a - T[a][b]['weight']
        sbo_b = sbo_b + T[a][b]['weight']
        Vwithoutab = [x for x in T.nodes() if x != a and x != b]
        while sbo_a >= sbo_b:
            x = Vwithoutab.pop()
            if T[a][x]['weight'] >= sbo_a - sbo_b + 1:  # max reverse possible more than still needed
                bo_DRS_by_b__value += sbo_a - sbo_b + 1  # done
                break
            bo_DRS_by_b__value += T[a][x]['weight']  # max reverse not enough
            sbo_a = sbo_a - T[a][x]['weight']
            if T[b][x]['weight'] >= sbo_a - sbo_b + 1:
                bo_DRS_by_b__value += sbo_a - sbo_b + 1
                break
            bo_DRS_by_b__value += T[b][x]['weight']
            sbo_b = sbo_b + T[b][x]['weight']
        return bo_DRS_by_b__value
