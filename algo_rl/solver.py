import uuid
import copy
import numpy as np
from collections import defaultdict

from algo_rl.util import init_cache, CacheItem
from algo_rl.policy import MixturePolicy
from algo_rl.util import calc_dist_1, calc_dist_2




def run(target1, target2, target3, proj_oracle=None, rl_oracle_generator=None, args=None):
    policy = MixturePolicy()
    ls_avg_measurement = []

    stats = defaultdict(int)
    cache = init_cache(rl_oracle_generator=rl_oracle_generator, args=args)

    init = True
    new_cache_item = True
    policy_id = 0

    value = float("inf")
    theta = proj_oracle.get_theta()

    converge = False


    for episode in range(args.num_epochs):
        print('episode-' + str(episode))

        min_value = float("inf")
        min_exp_rtn = None
        min_params = None
        min_uuid = None
        min_exp_TRAJS = []
        policy_state_dict = None

        # if init, or the cache was updated in last iteration(i.e., value<0), find the policy with the smallest value
        if value < 0 or np.isclose(0, value, atol=args.atol_cache, rtol=args.rtol_cache) or init:
            reset= True
            for item in cache:
                value = np.dot(theta, np.append(item.exp_rtn, args.mx_size))
                if value < min_value or init:
                    min_value = value
                    min_exp_rtn = item.exp_rtn
                    min_params = item.policy
                    min_uuid = item.uuid
                    min_exp_TRAJS = item.exp_TRAJS
                    init=False

            if min_value == float("inf"):
                min_params = rl_oracle.net.state_dict()
                min_exp_rtn = best_exp_rtn
                min_uuid = cache_uuid
                min_exp_TRAJS = exp_TRAJS


        # use min-value cached policy to warm start
        if reset:                                                     # if init, the cached was updated in last iteration
        # if min_value < 0 or np.isclose(0,min_value, atol=1e-1):     # only when this min-value < 0
            # print('HERE-1-warm_start')
            best_exp_rtn = min_exp_rtn
            cache_uuid = min_uuid
            exp_TRAJS = min_exp_TRAJS

            rl_oracle = rl_oracle_generator()
            new_params = rl_oracle.net.state_dict()
            new_params.update(min_params)
            rl_oracle.net.load_state_dict(new_params)
            rl_oracle.theta = theta[:-1]
            # rl_oracle.reset()
            stats['cache_calls'] += 1

            # update current policy state_dict (if this policy results in value<0, then this state_dict will be recorded)
            new_state_dict = {}
            for key in rl_oracle.net.state_dict():
                new_state_dict[key] = rl_oracle.net.state_dict()[key].clone()
            policy_state_dict = copy.deepcopy(new_state_dict)



        # Run RL Oracle to find a new policy
        else:
            # print('HERE-2-rl_oracle')
            rl_oracle.theta = theta[:-1]  # last element is artificial (makes the cone)
            #rl_oracle.reset()

            [best_exp_rtn, exp_TRAJS, paramm] = rl_oracle.learn_policy(n_traj=args.rl_traj,
                                                                       n_iter=args.rl_iter,
                                                                       cost=True,
                                                                       TRACK=ls_avg_measurement,
                                                                       entropy_check_number=args.entropy_check_number,
                                                                       entropy_coef=args.entropy_coef,
                                                                       entropy_threshod=args.entropy_threshold,
                                                                       entropy_increase=args.entropy_increase)

            # update current policy state_dict (if this policy results in value<0, then this state_dict will be recorded)
            policy_state_dict = copy.deepcopy(paramm)

            # update list of measurements, used to define entropy weights in rl_oracle
            ls_avg_measurement.append(best_exp_rtn)

            stats['oracle_calls'] += 1
            new_cache_item = True



        # check whether current 'best_exp_rtn' can achieve value<0 (i.e., whether current policy have good reward)
        reset = False
        value = np.dot(theta, np.append(best_exp_rtn, args.mx_size))

        # if current policy have value<0, meaning this 'positive response oracle' is finished, then move on to projection oracle
        if value < 0 or np.isclose(0, value, atol=args.atol_cache, rtol=args.rtol_cache):
            proj_oracle.update(best_exp_rtn.copy())
            theta = proj_oracle.get_theta()
            print("Update theta" + '\n')

            if new_cache_item:
                cache_uuid = uuid.uuid1()
                cache.append(CacheItem(best_exp_rtn, rl_oracle.net.state_dict(), cache_uuid, exp_TRAJS))
                new_cache_item = False

            policy.add_response(best_exp_rtn=best_exp_rtn, TRAJ_LIST=exp_TRAJS,
                                policy_state_dict=policy_state_dict, POLICY_ID=policy_id,
                                bins1=target1, bins2=target2, bins3=target3,
                                args=args, stats=stats)

            policy_id += 1

            dist_to_target = np.linalg.norm(policy.exp_rtn_of_avg_policy
                                            - proj_oracle.proj(policy.exp_rtn_of_avg_policy, args=args))

            status = "  dist-to-target: {}\n".format(dist_to_target)
            status += "  dist_to-prior_1: {}\n".format(calc_dist_1(policy.exp_rtn_of_avg_policy[:1], target1))
            status += "  dist_to-prior_2: {}\n".format(calc_dist_2(policy.exp_rtn_of_avg_policy[1:1+len(args.obs_states)], target2))
            status += "  dist_to-prior_3: {}\n".format(calc_dist_2(policy.exp_rtn_of_avg_policy[1+len(args.obs_states):], target3))

            # if current mixed-policy have average measurement have a very small distance to the constraint set, finish
            if (dist_to_target == 0.0):
                converge = True
                print('CONVERGE!')
                break

        # otherwise, have to return to rl oracle to find a new policy
        else:
            print("policy not good enough" + '\n')
            status = "   NO UPDATE, <theta,u> is equal: {}\n".format(value)
            #rl_oracle.reset()

        # print(status)


    if not (converge):
        print('REACH ITERATION LIMIT' + '\n')
    return policy






def print_results(target1, target2, target3, policy, args):
    dist_1 = calc_dist_1(policy.exp_rtn_of_avg_policy[:1], target1)
    if(int(dist_1[0])<0):
        print('c-1 is met')
    else:
        print('dist to c-1: ' + str(dist_1))

    dist_2 = calc_dist_2(policy.exp_rtn_of_avg_policy[1:1 + len(args.obs_states)], target2)
    print('dist to c-2: ' + str(dist_2))

    dist_3 = calc_dist_2(policy.exp_rtn_of_avg_policy[1 + len(args.obs_states):], target3)
    print('dist to c-3: ' + str(dist_3))




# record trajectories generated during the training process
def record_trajectories(policy, n_traj, target1, target2, target3, args, path):
    result = {}
    loss = []

    policy_id=[]
    single_measure_dist_1 = []
    single_measure_dist_2 = []
    single_measure_dist_3 = []
    avg_measure_dist_1 = []
    avg_measure_dist_2 = []
    avg_measure_dist_3 = []
    ls_traj=[]
    for i in range(n_traj):
        temp=[]
        ls_traj.append(temp)

    for p in policy.ls_trajectories:
        policy_id.append(p['policy_id'])

        single_dist1 = calc_dist_1(p['avg_measurement'][:1], target1)
        single_measure_dist_1.append(single_dist1)

        single_dist2 = calc_dist_2(p['avg_measurement'][1:1+len(args.obs_states)], target2)
        single_measure_dist_2.append(single_dist2)

        single_dist3 = calc_dist_2(p['avg_measurement'][1+len(args.obs_states):], target3)
        single_measure_dist_3.append(single_dist3)

        loss.append(p['avg_measurement'])
        avg_measure = np.average(np.stack(loss, axis=0), axis=0)

        dist1 = calc_dist_1(avg_measure[:1], target1)
        avg_measure_dist_1.append(dist1)

        dist2 = calc_dist_2(avg_measure[1:1+len(args.obs_states)], target2)
        avg_measure_dist_2.append(dist2)

        dist3 = calc_dist_2(avg_measure[1+len(args.obs_states):], target3)
        avg_measure_dist_3.append(dist3)

        for i in range(n_traj):
            ls_traj[i].append(p['traj_list']['traj'+str(i)])

    result['policy_id']=policy_id
    result['single_measure_dist_1'] = single_measure_dist_1
    result['single_measure_dist_2'] = single_measure_dist_2
    result['single_measure_dist_3'] = single_measure_dist_3
    result['avg_measure_dist_1'] = avg_measure_dist_1
    result['avg_measure_dist_2'] = avg_measure_dist_2
    result['avg_measure_dist_3'] = avg_measure_dist_3

    for i in range(n_traj):
        result['traj-'+str(i)]=ls_traj[i]

    f = open(path, 'w', encoding='utf-8')
    for k, v in result.items():
        f.write(k + '------------------------------------' + '\n')
        for value in v:
            s2 = str(value)
            f.write(s2 + '\n')
        f.write('\n')
    f.close()



