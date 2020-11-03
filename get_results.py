import pandas as pd
import numpy as np
from scipy.stats import entropy



def get_svf(trajs, all_trajs_number, nbstates, nb_traj_each_policy, path_1):

    def get_state_repr(sidx, nbstates):
        info = np.zeros(nbstates)
        info[sidx] = 1
        return info

    def get_svf_normalized(counts_info, nbstates):
        traffic_counts = {}
        for i in range(len(counts_info['state'])):
            key = counts_info['state'][i].strip('s')
            traffic_counts[str(key)] = int(counts_info['counts'][i])
        result = np.zeros(nbstates)
        sum = 0
        for k in traffic_counts.keys():
            result += (get_state_repr(int(k), nbstates) * traffic_counts[k])
            sum += traffic_counts[k]
        result /= sum
        for i in range(len(result)):
            if (result[i] == 0):
                result[i] = 1e-12
        svf = list(result)
        return svf


    def get_info(path, name):
        temp = []
        with open(path, "r") as f:
            flag = False
            for line in f.readlines():
                line = line.strip('\n')
                if not ('---' in line):
                    if (flag and line != ''):
                        temp.append(line)
                if ('---' in line):
                    flag = False
                    if (name in line):
                        flag = True
        return temp


    def get_count_predict(nbstates, nbtraj, input_path):
        info = {}
        info['policy_id'] = get_info(input_path, 'policy_id')
        for i in range(nbtraj):
            info['traj-' + str(i)] = get_info(input_path, 'traj-' + str(i))

        traj_list = []
        for i in range(len(info['policy_id'])):
            temp = {}
            temp['policy_id'] = str(info['policy_id'][i])
            for j in range(nbtraj):
                traj = []
                tinfo = info['traj-' + str(j)][i]
                for k in range(len(tinfo.split(',')) - 1):
                    traj.append(int(tinfo.split(',')[k][1:]))
                traj.append(int(tinfo.split(',')[-1][1:-1]))
                traj_list.append(traj)

        ls_count = np.zeros(nbstates)
        for traj in traj_list:
            for s in traj:
                if (s < nbstates):
                    ls_count[s] += 1
        ls_count = list(ls_count)

        ls_name = []
        for i in range(nbstates):
            ls_name.append('s' + str(i))

        dict = {'state': ls_name, 'counts': ls_count}
        return dict


    def find_traj_number(all_trajs_number, trajs):
        path_info_truth = []
        ls_key = list(all_trajs_number.keys())
        for k in ls_key:
            temp = {}
            temp['idx'] = k
            temp['path'] = trajs[k]
            temp['num'] = all_trajs_number[k]
            path_info_truth.append(temp)
        return path_info_truth



    def get_count_truth(all_trajs_number, trajs, nbstates):
        ls_states={}
        for i in range(nbstates):
            ls_states['s' + str(i)] = 0

        path_info = find_traj_number(all_trajs_number, trajs)

        ls_key = list(ls_states.keys())
        for k in ls_key:
            for info in path_info:
                for s in info['path']:
                    if('s' + str(s) == k):
                        ls_states[k] += info['num']
        ls_name=[]
        ls_count=[]
        for i in range(nbstates):
            ls_name.append('s' + str(i))
            ls_count.append(ls_states['s' + str(i)])
        dict={'state':ls_name, 'counts':ls_count}
        return dict


    counts_truth = get_count_truth(all_trajs_number, trajs, nbstates)
    svf_true = get_svf_normalized(counts_truth, nbstates)

    counts_predict = get_count_predict(nbstates, nbtraj=nb_traj_each_policy, input_path=path_1)
    svf_estimated = get_svf_normalized(counts_predict, nbstates)

    return svf_true, svf_estimated






def output_svf(svf_true, svf_estimated, args):
    result = {'state': [], 'svf_true': [], 'svf_estimated': [], 'KL_divergence_1': [], 'KL_divergence_2': []}
    for i in range(args.nb_state):
        result['state'].append('s' + str(i))
        result['svf_true'].append(svf_true[i])
        result['svf_estimated'].append(svf_estimated[i])
        result['KL_divergence_1'].append(entropy(svf_true, svf_estimated))
        result['KL_divergence_2'].append(entropy(svf_estimated, svf_true))

    pd.DataFrame(result).to_csv(args.output_2, index=False,
                                columns=['state', 'svf_true', 'svf_estimated', 'KL_divergence_1', 'KL_divergence_2'])




def output_estimation(estimation_result, args):
    output = {'unobs_state': [], 'flow_estimated': [], 'flow_true': []}
    for k in args.link_flow_true.keys():
        if not (k in list(args.link_flow_obs.keys())):
            output['unobs_state'].append(k)
            output['flow_estimated'].append(estimation_result['s'+k])
            output['flow_true'].append(args.link_flow_true[k])
    pd.DataFrame(output).to_csv(args.output_3, index=False,
                                columns=['unobs_state', 'flow_true', 'flow_estimated'])




