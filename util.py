import csv
import numpy as np
import pandas as pd


def load_network_properties(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    idx_3 = rows[0].index('nb_state')
    nb_state = int(rows[1][idx_3])
    idx_4 = rows[0].index('nb_action')
    nb_action = int(rows[1][idx_4])
    idx_5 = rows[0].index('nb_feature')
    nb_feature = int(rows[1][idx_5])
    return (nb_state, nb_action, nb_feature)




def load_origin_dest_states_info(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    origin_info = {}
    dest_info = {}
    idx_1 = rows[0].index('origin')
    idx_2 = rows[0].index('origin-states')
    idx_3 = rows[0].index('destination')
    idx_4 = rows[0].index('destination-states')
    for r in rows[1:]:
        info_ori = []
        temp_ori = r[idx_2].split(',')
        for i in range(len(temp_ori)-1):
            info_ori.append(int(temp_ori[i][1:]))
        info_ori.append(int(temp_ori[-1][1:-1]))
        origin_info[str(r[idx_1])] = info_ori

        info_des = []
        temp_des = r[idx_4].split(',')
        for i in range(len(temp_des)-1):
            info_des.append(int(temp_des[i][1:]))
        info_des.append(int(temp_des[-1][1:-1]))
        dest_info[str(r[idx_3])] = info_des
    return (origin_info, dest_info)




def load_trajectories(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    trajectories = {}
    trajs_number = {}
    trajs_number_truth = {}

    ls_traj = []
    ls_nb = []
    ls_nb_truth = []
    idx_1 = rows[0].index('ls_states')
    for i in range(1, len(rows)):
        ls_traj.append(rows[i][idx_1])
    idx_2 = rows[0].index('observed_nb')
    for i in range(1, len(rows)):
        ls_nb.append(int(rows[i][idx_2]))
    idx_3 = rows[0].index('ground_truth')
    for i in range(1, len(rows)):
        ls_nb_truth.append(int(rows[i][idx_3]))

    for i in range(len(ls_traj)):
        traj = ls_traj[i]
        t = []
        if(',' in traj):
            info = traj.split(',')
            t.append(int(info[0][1:]))
            for k in range(1, len(info) - 1):
                t.append(int(info[k]))
            t.append(int(info[-1][:-1]))
        else:
            t.append(int(traj[1:-1]))
        trajectories['t' + str(i)] = t
        trajs_number['t' + str(i)] = ls_nb[i]
        trajs_number_truth['t' + str(i)] = ls_nb_truth[i]
    return (trajectories, trajs_number, trajs_number_truth)




def get_state_feature(nb_state, nb_feature, ls_destination, path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    info = {}
    features = {}
    feature_length = 0

    for n in range(nb_feature):
        name = 'f' + str(n)
        feature_length += 1
        features[name]=[]
        idx = rows[0].index(name)
        for j in range(1, len(rows)):
            features[name].append(float(rows[j][idx]))

    for j in range(nb_state):
        info[str(j)] = []
        for a in range(nb_feature):
            name = 'f' + str(a)
            info[str(j)].append(features[name][j])

    for a in range(len(ls_destination)):
        info[str(nb_state + a)] = []
        for n in range(nb_feature):
            name = 'f' + str(n)
            info[str(nb_state + a)].append(0)

    return (feature_length, info)





def get_ls_origin(trajectories):
    ls_origin = []
    for k in trajectories.keys():
        if not (trajectories[k][0] in ls_origin):
            ls_origin.append(trajectories[k][0])
    ls_origin.sort()
    return ls_origin




def get_isd(trajectories, trajs_number, ls_origin):
    isd = {}
    for o in ls_origin:
        isd[str(o)] = 0
    sum = 0
    for k in trajectories.keys():
        sum += trajs_number[k]
        isd[str(trajectories[k][0])] += trajs_number[k]
    for k in isd.keys():
        isd[k] /= sum
    return isd



def get_ls_destination(trajectories):
    ls_destination = []
    for k in trajectories.keys():
        if not (trajectories[k][-1] in ls_destination):
            ls_destination.append(trajectories[k][-1])
    ls_destination.sort()
    return ls_destination




def get_dest_absorb_info(ls_destination, nb_state):
    dest_absorb_info = {}
    for i in range(len(ls_destination)):
        des = ls_destination[i]
        absorb = nb_state + i
        dest_absorb_info[str(des)] = absorb
    return dest_absorb_info




def find_origin(state, origin_info):
    info = 'n/a'
    for k in origin_info.keys():
        if(state in origin_info[k]):
            info = k
    return info

def find_destination(state, destination_info):
    info = 'n/a'
    for k in destination_info.keys():
        if(state in destination_info[k]):
            info = k
    return info


def get_od_pairs(trajectories, origin_info, destination_info):
    ls_od = []
    for k in trajectories.keys():
        start = find_origin(trajectories[k][0], origin_info)
        end = find_destination(trajectories[k][-1], destination_info)
        if not ([start, end] in ls_od):
            ls_od.append([start, end])
    od_pairs = {}
    for i in range(len(ls_od)):
        od_pairs['od-' + str(i)] = ls_od[i]
    return od_pairs




def get_maxlen(trajectories, od_pairs, origin_info, destination_info):
    od_len = {}
    for k in od_pairs.keys():
        od_len[k] = []

    def get_traj_len(traj):
        traj_len = 0
        for state in traj:
            traj_len += 1
        return traj_len

    for k in trajectories.keys():
        start = find_origin(trajectories[k][0], origin_info)
        end = find_destination(trajectories[k][-1], destination_info)
        idx = [a for a, b in od_pairs.items() if b == [start, end]]
        od_len[idx[0]].append(get_traj_len(trajectories[k]))

    od_maxlen = {}
    for k in od_pairs.keys():
        od_maxlen[k] = max(od_len[k])

    ls_len = []
    for k in od_maxlen.keys():
        ls_len.append(od_maxlen[k])

    return max(ls_len)




def get_distance_info(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    distance_info = {}
    idx_1 = rows[0].index('node_from')
    idx_2 = rows[0].index('node_to')
    idx_3 = rows[0].index('distance')
    i = 0
    for r in rows[1:]:
        distance_info[str(i)] = {'node_from':r[idx_1],
                                 'node_to':r[idx_2],
                                 'distance':float(r[idx_3])}
        i+=1
    return distance_info




def load_traffic_count_1(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    counts = {}
    ls_state = []
    ls_nb = []
    idx_1 = rows[0].index('obs_state')
    for i in range(1, len(rows)):
        ls_state.append(rows[i][idx_1])
    idx_2 = rows[0].index('observed_nb')
    for i in range(1, len(rows)):
        ls_nb.append(rows[i][idx_2])
    for i in range(len(ls_nb)):
        counts[str(ls_state[i])] = int(ls_nb[i])
    return counts




def load_traffic_count_2(path):
    data = pd.read_csv(path)
    counts_true = {}
    for i in range(len(data)):
        state = str(list(data['state'])[i])
        flow = int(list(data['flow'])[i])
        counts_true[state] = flow
    return counts_true





def traj_2_s_exp(ls_trajs, obs_states):
    freq_s = np.zeros(len(obs_states))
    for traj in ls_trajs:
        freq = np.zeros(len(obs_states))
        for s_idx in traj[:]:
            for i in range(len(obs_states)):
                if (s_idx == obs_states[i]):
                    freq[i] += 1
        freq_s += freq
    freq_s = freq_s / len(ls_trajs)
    return freq_s



def get_obs_state_prob(counts):
    obs_states = []
    for k in counts.keys():
        obs_states.append(int(k))

    all_trajs = []
    for k in counts.keys():
        for i in range(counts[k]):
            all_trajs.append([int(k)])

    obs_state_prob = list(traj_2_s_exp(all_trajs, obs_states))
    return (obs_states, obs_state_prob)


def traj_2_feature_expectation(ls_trajs, state_feature, feature_length):
    freq_s = np.zeros(feature_length)
    for traj in ls_trajs:
        freq = np.zeros(feature_length)
        for state in traj[:]:
            freq += state_feature[str(state)]
        freq_s += freq
    freq_s = freq_s / len(ls_trajs)
    return freq_s


def get_feature_expectation(trajectories, trajs_number, state_feature, feature_length):
    trajs = []
    for k in trajectories.keys():
        for i in range(trajs_number[k]):
            trajs.append(trajectories[k])
    feature_expectation = traj_2_feature_expectation(trajs, state_feature, feature_length)
    return feature_expectation



def get_transition_info(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    tran_info = {}
    idx_1 = rows[0].index('state_from')
    idx_2 = rows[0].index('state_to')
    idx_3 = rows[0].index('action')
    for r in rows[1:]:
        if not (str(r[idx_1]) in tran_info.keys()):
            tran_info[str(r[idx_1])] = {'action_next_state_pairs': [[int(r[idx_3]), int(r[idx_2])]]}
        else:
            tran_info[str(r[idx_1])]['action_next_state_pairs'].append([int(r[idx_3]), int(r[idx_2])])
    return tran_info