from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


actors_sequence =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

cbim = [9, 5, 2, 1, 7, 8 , 10, 3, 4, 11, 6]
cim = [11, 7, 3, 2, 8, 4 , 9 , 5 , 6, 10, 1]
deg_c = [11, 3, 5, 1, 2, 6, 9, 10, 4, 7, 8]
deg_c_d = [10, 4, 6, 1, 2, 11, 7 , 8 , 3, 5 , 9]
k_sh = [11, 3, 1, 2, 5, 8, 10, 6, 7, 9, 4]
k_sh_m = [11, 3, 4, 1, 2, 5, 9, 10, 6, 7, 8]
kpp_sh = [8, 6, 3, 7, 2, 10, 11, 1 , 4, 9 , 5]
nghb_2s = [7, 1, 6, 5, 8, 2, 9, 11, 3, 10, 4]
nghb_1s = [9, 1, 4, 3, 5, 2, 10, 11, 6, 7, 8]
nghb_sd = [7, 1, 8, 3, 4, 2 , 9 , 10, 5, 6 , 11]
p_rnk = [11, 6, 2, 3, 7, 10, 9, 5, 4, 8, 1]
p_rnk_m = [11, 1, 6, 3, 5, 2, 10, 9, 4, 7, 8]
v_rnk = [8, 6, 4, 1, 7, 9, 11, 5, 2, 10, 3]
v_rnk_m = [10, 1, 9, 5, 6, 2, 7, 8, 4, 3, 11]


def positions_to_ranking(pos_list: list[int], ids_list: list[int]) -> list[int]:
    arr = [-1] * len(pos_list)
    for id in ids_list:
        place = pos_list[id - 1]
        arr[place - 1] = id
    return arr


all_ranks = {
    "cbim": positions_to_ranking(cbim, actors_sequence),
    "cim": positions_to_ranking(cim, actors_sequence),
    "deg_c": positions_to_ranking(deg_c, actors_sequence),
    "deg_c_d": positions_to_ranking(deg_c_d, actors_sequence),
    "k_sh": positions_to_ranking(k_sh, actors_sequence),
    "k_sh_m": positions_to_ranking(k_sh_m, actors_sequence),
    "kpp_sh": positions_to_ranking(kpp_sh, actors_sequence),
    "nghb_2s": positions_to_ranking(nghb_2s, actors_sequence),
    "nghb_1s": positions_to_ranking(nghb_1s, actors_sequence),
    "nghb_2s": positions_to_ranking(nghb_2s, actors_sequence),
    "p_rnk": positions_to_ranking(p_rnk, actors_sequence),
    "p_rnk_m": positions_to_ranking(p_rnk_m, actors_sequence),
    "v_rnk": positions_to_ranking(v_rnk, actors_sequence),
    "v_rnk_m": positions_to_ranking(v_rnk_m, actors_sequence),
}

print(len(all_ranks))
print(len(list(combinations(all_ranks, 2))))


def jaccard_additive(ranking_a: list[int], ranking_b: list[int]) -> list[float]:
    jaccards = []
    for cutoff in range(1, len(ranking_a)+1):
        seed_set_a = set(ranking_a[:cutoff])
        seed_set_b = set(ranking_b[:cutoff])
        jaccard = len(seed_set_a.intersection(seed_set_b)) / len(seed_set_a.union(seed_set_b))
        jaccards.append(jaccard)
    return jaccards


jaccards_list = []
for (rank_a, rank_b) in combinations(all_ranks, 2):
    if rank_a == rank_b:
        continue
    print(rank_a, rank_b)
    jaccards_list.append(jaccard_additive(all_ranks[rank_a], all_ranks[rank_b]))
jaccards_arr = np.array(jaccards_list)

jaccard_mean = jaccards_arr.mean(axis=0)
jaccard_std = jaccards_arr.std(axis=0)
budgets = np.ceil(100 * np.arange(start=1, stop=len(actors_sequence)+1) / len(actors_sequence)).astype(int)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(budgets, jaccard_mean, color="seagreen")
ax.fill_between(budgets, jaccard_mean - jaccard_std, jaccard_mean + jaccard_std, color="seagreen", alpha=0.2)
ax.set_xlim(min(budgets), max(budgets))
ax.set_ylim(0, 1)
ax.set_xlabel("Budget size [%]")
ax.set_ylabel("Jaccard similarity")
fig.set_size_inches(5, 3)
fig.tight_layout()
# plt.show()
fig.savefig("seed_sets_toy_net.pdf", dpi=300)
