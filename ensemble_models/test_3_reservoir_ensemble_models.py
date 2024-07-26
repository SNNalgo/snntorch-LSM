import numpy as np
from ensemble_3_reservoir_long_short_dist_Gabor_lsm import long_short_Gabor_ensemble_lsm
from ensemble_lsm import simple_ensemble_lsm

in_conns_gabor = [0.05]
in_conns_simple = [0.15]

params_gabor = []
params_simple = []

scores_gabor = []
scores_simple = []

reps = 4
num_res = 3

#long_dist1 = 5
#long_dist2 = 10

long_dist1 = 4
long_dist2 = 6

for i in range(reps):
    print('running gabor + long-short ensemble, rep ', i+1)
    for in_conn_gabor in in_conns_gabor:
        score = long_short_Gabor_ensemble_lsm(in_conn_gabor, long_dist1=long_dist1, long_dist2=long_dist2)
        params_gabor.append((in_conn_gabor, long_dist1, long_dist2))
        scores_gabor.append(score)
    print('running simple ensemble, rep ', i+1)
    for in_conn_simple in in_conns_simple:
        score = simple_ensemble_lsm(in_conn_simple, num_res=num_res)
        params_simple.append(in_conn_simple)
        scores_simple.append(score)

params_gabor_np = np.array(params_gabor)
scores_gabor_np = np.array(scores_gabor)

params_simple_np = np.array(params_simple)
scores_simple_np = np.array(scores_simple)

save_path = './'
np.savez(save_path + 'ensemble_comparisons_3_reservoir_2', params_gabor=params_gabor_np, scores_gabor=scores_gabor_np, params_simple=params_simple_np, scores_simple=scores_simple_np)

exp_log = np.load('ensemble_comparisons_3_reservoir_2.npz')
print(list(exp_log.keys()))
print('params simple: ', exp_log['params_simple'])
print('scores simple: ', exp_log['scores_simple'])
print('params gabor: ', exp_log['params_gabor'])
print('scores gabor: ', exp_log['scores_gabor'])