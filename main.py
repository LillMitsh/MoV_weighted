#!/usr/bin/env python

'''
    File:           weightedMoV_experiments.py
    Master thesis:  Margin of Victory for Weighted Tournament Solutions
    Author:         Michelle Luise Döring (m.doering@tu-berlin.de)
    Date:           May 26, 2022
    Credit:         Markus Brill, Ulrike Schmidt-Kraepelin, Warut Suksompong
                    --Most of the main file is taken from https://github.com/uschmidtk/MoV
                    --Profile generations functions are slightly adjusted to generate weighted tournaments instead of unweighted
                    --Main run is slightly adjusted to gather more data, fit weighted tournaments and call the new functions
'''

import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random

import generate_profiles as pl

from functions_borda import *
from functions_splitcycle import *
from functions_wuncoveredset import *


############################################################
# -------------------- CALCULATE MoVs -------------------- #
############################################################

def movs(T, n, bo=True, sc=True, wuc=True):
    """ Computes the MoV for all alternatives in the tournament T for the tournament solutions set to TRUE

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament
    m:  int
        number of alternatives of T
    n:  int
        number of matches/voters of T
    bo/sc/wuc: truth value
        a candidate

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """

    movs = pd.DataFrame()  # Two-dimensional, size-mutable, potentially heterogeneous tabular data

    mgg = create_margin_graph(T)

    for v in T.nodes():
        if bo:
            movs.at[v, 'bo_mov'] = bo_mov(T, v)
            if movs.at[v, 'bo_mov'] == -1:
                movs.at[v, 'bo_winner'] = 0
            else:
                movs.at[v, 'bo_winner'] = 1
        if sc:
            movs.at[v, 'sc_mov'] = sc_mov(T, mgg, v, n)
            if movs.at[v, 'sc_mov'] == -1:
                movs.at[v, 'sc_winner'] = 0
            else:
                movs.at[v, 'sc_winner'] = 1
        if wuc:
            movs.at[v, 'wuc_mov'] = wuc_mov(T, v, n)
            if movs.at[v, 'wuc_mov'] == -1:
                movs.at[v, 'wuc_winner'] = 0
            else:
                movs.at[v, 'wuc_winner'] = 1
    return movs


####################################################
################# CREATE TOURNAMENTS ###############
####################################################

def random_tournament_add_weights(T, n):
    for e in T.edges():
        u, v = e[0], e[1]
        w = random.randrange(0, n, 1)
        T.add_edge(u, v, weight=w)
        T.add_edge(v, u, weight=n - w)
        T.add_edge(u, v, margin=w - (n - w))
        T.add_edge(v, u, margin=(n - w) - w)


def condorcet_tournament(n, m, p):
    all_edges = []

    for i in range(n):
        for s in itertools.combinations(range(m), 2):
            s = list(s)
            #print("s in round ",i," is ",s)
            if s[0] != s[1]:
                coin = np.random.rand()
                if ((s[0] < s[1]) and (coin <= p)) or ((s[1] < s[0]) and (coin > p)):
                    all_edges.append((s[0], s[1]))
                else:
                    all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()
    T = nx.DiGraph()
    T.add_nodes_from(range(m))

    for (u, v) in list(edge_count.index):
        w = edge_count[(u, v)]
        T.add_edge(u, v, weight= w)
        T.add_edge(u, v, margin= w - (n - w))
        T.add_edge(v, u, weight= n - w)
        T.add_edge(v, u, margin=(n - w) - w)
    return T

def condorcet_tournament_direct(m,p):
    all_edges=[]
    for s in itertools.combinations(range(m),2):
        s = list(s)
        if s[0]!=s[1]:
            coin = np.random.rand()
            if ((s[0] < s[1]) and (coin <= p)) or ((s[1] < s[0]) and (coin > p)):
                all_edges.append((s[0],s[1]))
            else:
                all_edges.append((s[1],s[0]))
    T = nx.DiGraph()
    T.add_nodes_from(range(m))
    T.add_edges_from(all_edges)

    return T

def impartial_culture(n, m):
    all_edges = []
    for i in range(n):
        order = list(np.random.permutation(range(m)))
        for s in itertools.combinations(range(m), 2):
            s = list(s)
            if s[0] != s[1]:
                if (order.index(s[0]) < order.index(s[1])):
                    all_edges.append((s[0], s[1]))
                else:
                    all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()

    T = nx.DiGraph()
    T.add_nodes_from(range(m))

    for (u, v) in list(edge_count.index):
        w = edge_count[(u, v)]
        T.add_edge(u, v, weight=w)
        T.add_edge(u, v, margin=w - (n - w))
        T.add_edge(v, u, weight= n - w)
        T.add_edge(v, u, margin=(n - w) - w)
    return T


def mallows(n, m, phi):
    candmap = {i: i for i in range(m)}
    rankmapcounts = pl.gen_mallows(n, candmap, [1], [phi], [list(range(m))])
    all_edges = []
    for i in range(len(rankmapcounts[1])):
        for s in itertools.combinations(range(m), 2):
            if s[0] != s[1]:
                if rankmapcounts[0][i][s[0]] < rankmapcounts[0][i][s[1]]:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[0], s[1]))
                else:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()

    T = nx.DiGraph()
    T.add_nodes_from(range(m))

    for (u,v) in list(edge_count.index):
        w = edge_count[(u,v)]
        T.add_edge(u,v,weight=w)
        T.add_edge(u,v,margin= w-(n-w))
        T.add_edge(v, u, weight= n - w)
        T.add_edge(v, u, margin=(n - w) - w)
    return T


def urn(n, m, replace):
    candmap = {i: i for i in range(m)}
    rankmapcounts = pl.gen_urn_strict(n, replace, candmap)

    all_edges = []
    for i in range(len(rankmapcounts[1])):
        for s in itertools.combinations(range(m), 2):
            if s[0] != s[1]:
                if rankmapcounts[0][i][s[0]] < rankmapcounts[0][i][s[1]]:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[0], s[1]))
                else:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()

    T = nx.DiGraph()
    T.add_nodes_from(range(m))

    for (u,v) in list(edge_count.index):
        w = edge_count[(u,v)]
        T.add_edge(u,v,weight=w)
        T.add_edge(u,v,margin= w-(n-w))
        T.add_edge(v, u, weight= n - w)
        T.add_edge(v, u, margin=(n - w) - w)
    return T


def create_margin_graph(T):
    """Create margin graph of T

    Parameters
    ----------
    T:  nx.Digraph
        n-weighted tournament

    Key assumptions:
        * every edge of T has an attribute called 'weight' and maybe even 'margin'
    """
    mgg = nx.DiGraph()
    mgg.add_nodes_from(T.nodes())
    for (u, v) in T.edges():
        if T[u][v]['margin'] > 0:
            mgg.add_edge(u, v, weight=T[u][v]['margin'])
    return mgg


####################################################
################# START EXPERIMENTS ################
####################################################
tournament_model_list = ['random','condorcet_direct','condorcet_voters','impartial','mallows','urn'] #full list of used models
#tournament_model_list = ['condorcet_direct'] #full list of used models
#tournament_model_list = ['random']  # recommended for first test run

# tournament_sample_size = 100 #original sample size
tournament_sample_size = 100  # recommended size for first test run

# m_list = [5,10,15,20,25,30] #list from original experiments    --number of alternatives
m_list = [5,10,15,20,25,30]  # recommended list for first test run 3,5,6

n_list = [2, 10, 51, 100]
n = 100  # --number of voter

run = True  # decides whether experiments run
plot_bol = False    # decides whether plots are produced
zoomin = True  # if set to true produces seperate plots for each size of n,target value and model

title = {'random': 'Uniform Random', 'condorcet_direct': 'Condorcet Noise p=0.55', 'condorcet_voters': 'Condorcet Noise via n Voters p=0.55',
         'impartial': 'Impartial Culture', 'mallows': 'Mallows (phi = 0.95)', 'urn': 'Urn (alpha=10)'}

if run:
    ''' Create Tournaments: For every chosen generation model in tournament_model_list and for each possible number of alternatives in m_list |tournament_sample_size| many '''
    for model in tournament_model_list:
        print("----------------------------------------------MODEL----------------------------------------------:\t",model,"----------------------------------------------")
        '''Store interesting data in Panda Data Frames '''
        count_alternatives_with_max_mov_value = pd.DataFrame(columns=['i', 'n'])
        count_alternatives_with_max_mov_value = count_alternatives_with_max_mov_value.set_index(['i',
                                                                                                 'n'])  # Set the DataFrame index (row labels) using one or more existing columns or arrays (of the correct length). The index can replace the existing index or expand on it
        count_number_of_unique_mov_values = pd.DataFrame(columns=['i', 'n'])
        count_number_of_unique_mov_values = count_number_of_unique_mov_values.set_index(['i', 'n'])
        value_of_max_mov_value = pd.DataFrame(columns=['i', 'n'])
        value_of_max_mov_value = value_of_max_mov_value.set_index(['i', 'n'])

        avg_value_of_max_mov_value = pd.DataFrame(columns=['i', 'n'])
        avg_value_of_max_mov_value = avg_value_of_max_mov_value.set_index(['i', 'n'])

        for m in m_list:
            for i in range(tournament_sample_size):
                if model == 'random':
                    T = nx.algorithms.tournament.random_tournament(m)
                    random_tournament_add_weights(T, n)
                if model == 'condorcet_voters':
                    T = condorcet_tournament(n, m, 0.55)
                if model == 'condorcet_direct':
                    T=condorcet_tournament_direct(m,0.55)
                    print(T.edges(data=True))
                    random_tournament_add_weights(T, n)
                    print(T.edges(data=True))
                if model == 'impartial':
                    T = impartial_culture(n, m)
                if model == 'mallows':
                    T = mallows(n, m, 0.95)
                if model == 'urn':
                    T = urn(n, m, 10)

                ''' Compute the MoV values for the tournament T '''
                mov_data = movs(T, n)
                # print(mov_data)
                # print("mov_data.columns:",mov_data.columns)
                # print("list(mov_data.columns):",list(mov_data.columns))

                ''' Collect interesting Data '''
                for col in list(mov_data.columns):
                    count_alternatives_with_max_mov_value.at[(i, m), col] = len(
                        mov_data[col][mov_data[col] == mov_data[col].max()])
                for col in ['bo_mov', 'sc_mov', 'wuc_mov']:
                    if col in list(mov_data.columns):
                        count_number_of_unique_mov_values.at[(i, m), col] = len(
                            [i for i in mov_data[col].unique() if i > 0])
                        value_of_max_mov_value.at[(i, m), col] = mov_data[col].max()
                print("table of everything with i=", i, "n=", n, "m=", m, "\n", mov_data,"----------------------------------------------",model)

            if zoomin:
                for col in list(count_alternatives_with_max_mov_value.columns):
                    All = slice(None)
                    maaxi = count_alternatives_with_max_mov_value.loc[(All, m), col]
                    plt.figure()
                    maaxi.plot(kind='hist', bins=m,
                               title=col + ' - How many alternatives with max MoV - ' + title[model], density=1)
                    # 'Histogram of Maximum Equivalence Class - ' + title[model],density=1)
                    plt.savefig('experiment_figures/' + str(m) + '-' + col + '-hist-max-' + model)
                    plt.close()

                for col in ['bo_mov', 'sc_mov', 'wuc_mov']:
                    unique2 = count_number_of_unique_mov_values.loc[(All, m), col]
                    plt.figure()
                    unique2.plot(kind='hist', bins=n, title=col + ' - How many unique MoV - ' + title[model], density=1)
                    # 'Histogram of count_number_of_unique_mov_values Values - ' + title[model],density=1)
                    plt.savefig(
                        'experiment_figures/' + str(m) + '-' + col + '-hist-count_number_of_unique_mov_values-' + model)
                    plt.close()

        print("count_alternatives_with_max_mov_value\n", count_alternatives_with_max_mov_value)  # how many with max MoV
        print("count_number_of_unique_mov_values\n",
              count_number_of_unique_mov_values)  # how many count_number_of_unique_mov_values MoVs
        print("value_of_max_mov_value\n", value_of_max_mov_value)

        count_alternatives_with_max_mov_value.to_csv('experiment_data/' + 'max-' + model + '.csv')
        # count_number_of_unique_mov_values.to_csv('experiment_data/'+'unique-' + model + '.csv')
        value_of_max_mov_value.to_csv('experiment_data/' + 'max-value' + model + '.csv')

        avg_value_of_max_mov_value = value_of_max_mov_value.groupby('n').mean()
        print("avg_value_of_max_mov_value\n", avg_value_of_max_mov_value)
        avg_value_of_max_mov_value.to_csv('experiment_data/' + 'avg_mov_value-' + model + '.csv')
        avg_count_alternatives_with_max_mov_value = count_alternatives_with_max_mov_value.groupby('n').mean()
        print("avg_count_alternatives_with_max_mov_value\n", avg_count_alternatives_with_max_mov_value)
        avg_count_alternatives_with_max_mov_value.to_csv('experiment_data/' + 'agg_max-' + model + '.csv')

        count_number_of_unique_mov_values.to_csv(
            'experiment_data/' + 'count_number_of_unique_mov_values-' + model + '.csv')
        avg_count_number_of_unique_mov_values = count_number_of_unique_mov_values.groupby('n').mean()
        print("avg_count_number_of_unique_mov_values\n", avg_count_number_of_unique_mov_values)
        avg_count_number_of_unique_mov_values.to_csv(
            'experiment_data/' + 'avg_count_number_of_unique_mov_values-' + model + '.csv')

        if plot_bol:
            ind_str = np.array(['n = ' + str(x) for x in n_list])
            ind = np.arange(len(n_list))
            width = 7 / 100
            fig, ax = plt.subplots()
            hfont = {'fontname': 'Times'}

            if 'bo_mov' in list(mov_data.columns):
                rects1 = ax.bar(ind - 0.33, np.array(avg_count_alternatives_with_max_mov_value['bo_mov']), width,
                                label='BO-mov', color='#2961A6')
                rects2 = ax.bar(ind - 0.33 + width, np.array(avg_count_alternatives_with_max_mov_value['bo_winner']), width,
                                label='bo_winner', color='#76A8D7')
            if 'sc_mov' in list(mov_data.columns):
                rects3 = ax.bar(ind - 0.33 + 24 / 3 * width, np.array(avg_count_alternatives_with_max_mov_value['sc_mov']),
                                width, label='SC-mov', color='#E5AB14')
                rects4 = ax.bar(ind - 0.33 + 27 / 3 * width,
                                np.array(avg_count_alternatives_with_max_mov_value['sc_winner']), width, label='sc_winner',
                                color='#F7E091')
            if 'wuc_mov' in list(mov_data.columns):
                rects5 = ax.bar(ind - 0.33 + 8 / 3 * width, np.array(avg_count_alternatives_with_max_mov_value['wuc_mov']),
                                width, label='wUC-mov', color='#B02417')
                rects6 = ax.bar(ind - 0.33 + 11 / 3 * width,
                                np.array(avg_count_alternatives_with_max_mov_value['wuc_winner']), width,
                                label='wuc_winner', color='#F19393')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            # ax.set_ylabel('Average Size')
            ax.set_title(title[model], loc='right', size=14, fontname='Times')
            ax.set(ylim=[0, max(n_list) + 1])
            ax.set_xticks(ind)
            ax.set_xticklabels(ind_str)
            ax.tick_params(labelsize=13)
            ax.minorticks_on()
            ax.legend(loc='upper left')
            plt.savefig('experiment_figures/' + 'max-' + model + '2', dpi=500)
            plt.close()

            width = 0.14
            fig, ax = plt.subplots()
            if 'bo_mov' in list(mov_data.columns):
                rects1 = ax.bar(ind - 0.27, np.array(avg_count_number_of_unique_mov_values['bo_mov']), width,
                                label='BO-mov', color='#2961A6')
            if 'sc_mov' in list(mov_data.columns):
                rects3 = ax.bar(ind - 0.27 + 3 * width, np.array(avg_count_number_of_unique_mov_values['sc_mov']), width,
                                label='SC-mov', color='#E5AB14')
            if 'wuc_mov' in list(mov_data.columns):
                rects7 = ax.bar(ind - 0.27 + width, np.array(avg_count_number_of_unique_mov_values['wuc_mov']), width,
                                label='wUC-mov', color='#B02417')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            # ax.set_ylabel('Average Number')
            ax.set_title(title[model], loc='right', size=14, fontname='Times')
            ax.set(ylim=[0, 21])
            ax.set_xticks(ind)
            ax.set_xticklabels(ind_str)
            ax.tick_params(labelsize=13)
            ax.minorticks_on()
            ax.legend(loc='upper left')
            plt.savefig('experiment_figures/' + 'count_number_of_unique_mov_values-' + model + '2', dpi=500)
            plt.close()
