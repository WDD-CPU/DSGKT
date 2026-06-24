import math
import torch
import numpy as np

# Global parameters, adjusted according to different datasets
skill_num = 102
dymaic_length = 5

from collections import defaultdict


def emotional_vector_calculation(sequences, skill_num, max_len):
    students_all = sequences

    students_split_data = []
    split_seqs_for_cluster = []
    studentids_set = set()
    max_seg = 0
    max_skills = 0
    max_attempts = 0

    for seq in sequences:
        stu_id = seq['student_id']
        studentids_set.add(stu_id)

        exercise_seq = seq['exercise_seq']
        skill_seq = seq['skill_seq']
        response_seq = seq['response_seq']
        time_seq = seq['time_seq']
        interval_seq = seq['interval_seq']
        hint_seq = seq['hint_seq']
        attempt_seq = seq['attempt_seq']
        start_time_seq = seq['start_time_seq']
        mask_seq = seq['mask_seq']

        n = len(skill_seq)
        if n == 0:
            continue

        n_split = (n + max_len - 1) // max_len
        max_seg = max(max_seg, n_split)
        if n > 0:
            max_skills = max(max_skills, skill_seq.max())
        max_attempts = max(max_attempts, n)

        for k in range(n_split):
            start = k * max_len
            end = min((k + 1) * max_len, n)

            pro = exercise_seq[start:end]
            q = skill_seq[start:end]
            a = response_seq[start:end]
            at = time_seq[start:end]
            st = start_time_seq[start:end]
            ac = attempt_seq[start:end]
            hc = hint_seq[start:end]
            it = interval_seq[start:end].copy()
            ms = mask_seq[start:end]

            tuple_data = (
                [stu_id, k],
                n_split,
                pro, q, a, at, st, ac, hc, it, ms
            )
            students_split_data.append(tuple_data)

            split_seq = {
                'student_id': stu_id,
                'exercise_seq': pro,
                'skill_seq': q,
                'response_seq': a,
                'time_seq': at,
                'interval_seq': it,
                'hint_seq': hc,
                'attempt_seq': ac,
                'start_time_seq': st,
                'mask_seq': ms,
                'subseq_idx': k,
                'total_subseqs': n_split,
                'original_seq_length': n,
                'subseq_start_idx': start,
                'subseq_end_idx': end,
                'subseq_length': len(pro),
                'is_last_subseq': (k == n_split - 1)
            }
            split_seqs_for_cluster.append(split_seq)

    studentids = sorted(studentids_set)

    if len(studentids) == 0:
        return [], [], [], [], 0, 0, 0, []

    all_intervals = []
    for s in split_seqs_for_cluster:
        st_list = s['start_time_seq']
        if len(st_list) > 1:
            intervals = [st_list[i] - st_list[i - 1] for i in range(1, len(st_list))]
            all_intervals.extend(intervals)
    global_st_interval = np.mean(all_intervals) if all_intervals else 0.0

    question_dict = defaultdict(lambda: {'total': 0, 'correct': 0})
    at_dict = defaultdict(lambda: {'total_time': 0, 'count': 0})
    ac_dict = defaultdict(lambda: {'total_count': 0, 'count': 0})

    all_q, all_a, all_at, all_st, all_ac = [], [], [], [], []
    for s in split_seqs_for_cluster:
        all_q.extend(s['skill_seq'].tolist())
        all_a.extend(s['response_seq'].tolist())
        all_at.extend(s['time_seq'].tolist())
        all_st.extend(s['start_time_seq'].tolist())
        all_ac.extend(s['attempt_seq'].tolist())

    for q, a, at, st, ac in zip(all_q, all_a, all_at, all_st, all_ac):
        question_dict[q]['total'] += 1
        if a == 1:
            question_dict[q]['correct'] += 1
        at_dict[q]['total_time'] += at
        at_dict[q]['count'] += 1
        ac_dict[q]['total_count'] += ac
        ac_dict[q]['count'] += 1

    new_dict = {}
    for q in question_dict:
        total = question_dict[q]['total']
        correct = question_dict[q]['correct']
        total_time = at_dict[q]['total_time']
        count_at = at_dict[q]['count']
        total_ac = ac_dict[q]['total_count']
        count_ac = ac_dict[q]['count']

        correct_ratio = correct / total if total > 0 else 0
        time_ratio = total_time / count_at if count_at > 0 else 0
        ac_ratio = total_ac / count_ac if count_ac > 0 else 0

        new_dict[q] = {
            'correct_ratio': correct_ratio,
            'time_ratio': time_ratio,
            'ac_ratio': ac_ratio
        }

    max_stu = max(studentids) + 1
    xtotal = np.zeros((max_stu, skill_num))
    x1 = np.zeros((max_stu, skill_num))
    x4 = np.zeros((max_stu, skill_num))
    x8 = np.zeros((max_stu, skill_num))
    x0 = np.zeros((max_stu, skill_num))
    x3 = np.zeros((max_stu, skill_num))
    x7 = np.zeros((max_stu, skill_num))

    cluster_data = []
    for s in split_seqs_for_cluster:
        stu_id = s['student_id']
        seg_id = s['subseq_idx']
        total_subseqs = s['total_subseqs']
        q_list = s['skill_seq']
        a_list = s['response_seq']
        at_list = s['time_seq']
        st_list = s['start_time_seq']
        ac_list = s['attempt_seq']

        relative_participation_rate = total_subseqs / max_seg if max_seg > 0 else 1.0

        if len(st_list) > 1:
            local_intervals = [st_list[i] - st_list[i - 1] for i in range(1, len(st_list))]
            local_avg_interval = np.mean(local_intervals)
        else:
            local_avg_interval = 0.0
        interval_diff = local_avg_interval - global_st_interval

        xtotal[stu_id] = np.zeros(skill_num)
        x1[stu_id] = np.zeros(skill_num)
        x4[stu_id] = np.zeros(skill_num)
        x8[stu_id] = np.zeros(skill_num)

        for q, a, at, ac in zip(q_list, a_list, at_list, ac_list):
            xtotal[stu_id, q] += 1
            x4[stu_id, q] += at
            x8[stu_id, q] += ac
            x0[stu_id, q] = new_dict.get(q, {'correct_ratio': 0})['correct_ratio']
            x3[stu_id, q] = new_dict.get(q, {'time_ratio': 0})['time_ratio']
            x7[stu_id, q] = new_dict.get(q, {'ac_ratio': 0})['ac_ratio']
            if a == 1:
                x1[stu_id, q] += 1

        xsr = [x / max(y, 1e-6) for x, y in zip(x1[stu_id], xtotal[stu_id])]
        xfr = [x for x in x0[stu_id]]
        x4r = [y / max(x, 1e-6) if x != 0 else 0 for x, y in zip(xtotal[stu_id], x4[stu_id])]
        x3r = [x for x in x3[stu_id]]
        x8r = [x / max(y, 1e-6) for x, y in zip(x8[stu_id], xtotal[stu_id])]
        x7r = [x for x in x7[stu_id]]

        diff_correct = np.nan_to_num(np.array(xsr) - np.array(xfr))
        diff_time = np.nan_to_num(np.array(x4r) - np.array(x3r))
        diff_attempt = np.nan_to_num(np.array(x8r) - np.array(x7r))

        def merge_datasets(d1, d2, d3):
            d1 = np.expand_dims(d1, 1)
            d2 = np.expand_dims(d2, 1)
            d3 = np.expand_dims(d3, 1)
            return np.concatenate((d1, d2, d3), axis=0)

        merged = merge_datasets(diff_correct, diff_time, diff_attempt)
        merged = np.append(merged, relative_participation_rate)
        merged = np.append(merged, interval_diff)
        merged = np.append(merged, stu_id)
        merged = np.append(merged, seg_id)
        cluster_data.append(merged)

    return students_all, students_split_data, cluster_data, studentids, max_skills, max_seg, max_attempts, split_seqs_for_cluster


def is_zero_set(s):
    return s == {0}


def euclideanDistance(instance1, instance2, select_kc):
    distance = 0
    new_select_kc = [num + i * skill_num for num in select_kc for i in range(3)]

    for x in new_select_kc:
        if x < len(instance1) and x < len(instance2):
            distance += pow((instance1[x] - instance2[x]), 2)

    return math.sqrt(distance)


def dymaic_cluster_std(students_split_data):
    kc_set_list_divid = []

    for item in students_split_data:
        q_seq = item[3]

        KC_set_num = set()
        count_num = 1
        groups_for_this_subseq = []

        for skill_id in q_seq:
            if count_num <= dymaic_length:
                KC_set_num.add(int(skill_id))
                count_num += 1
            else:
                if KC_set_num:
                    groups_for_this_subseq.append(KC_set_num.copy())

                KC_set_num = {int(skill_id)}
                count_num = 2

        if KC_set_num:
            groups_for_this_subseq.append(KC_set_num)

        kc_set_list_divid.append(groups_for_this_subseq)

    return kc_set_list_divid


def k_means_clust(
        train_students_split, val_students_split, test_students_split,
        train_cluster_data, val_cluster_data, test_cluster_data,
        max_stu, max_seg,
        num_clust=4,
        num_iter=102
):
    identifiers = 2
    feature_dims = 3 * num_iter + 2
    max_stu = int(max_stu)
    max_seg = int(max_seg)
    cluster = {}

    data = []
    kc_set_list_divid_train = dymaic_cluster_std(train_students_split)
    kc_set_list_divid_val = dymaic_cluster_std(val_students_split)
    kc_set_list_divid_test = dymaic_cluster_std(test_students_split)

    for ind, i in enumerate(train_cluster_data):
        data.append(i[:feature_dims])

    data_np = np.array(data)
    data = torch.from_numpy(data_np).float()
    data = torch.from_numpy(np.array(data)).float()

    centroids = data[torch.randperm(len(data))[:num_clust]]

    for iteration in range(num_iter):
        distances = ((data[:, None] - centroids[None, :]) ** 2).sum(-1)
        indices = distances.argmin(1)
        clusters = [data[indices == i] for i in range(num_clust)]
        new_centroids = torch.stack([c.mean(0) for c in clusters])

        if torch.allclose(centroids, new_centroids, rtol=1e-4):
            print(f"Emotion clustering converged at iteration {iteration + 1} based on training set with random centroid initialization and Euclidean distance optimization.")
            break
        centroids = new_centroids

    print("Starting emotion cluster assignment for training set...")
    for ind, i in enumerate(train_cluster_data):
        inst = torch.Tensor(i[:feature_dims])
        closest_clusts = []

        q_seq = train_students_split[ind][3]
        total_questions = len(q_seq)

        for virtual_std in range(len(kc_set_list_divid_train[ind])):
            select_kc = kc_set_list_divid_train[ind][virtual_std]
            min_dist = float('inf')
            closest_clust = None

            for j in range(num_clust):
                cur_dist = euclideanDistance(inst, centroids[j], select_kc)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = j

            closest_clusts.append(closest_clust)

        key = (int(i[-2]), int(i[-1]))
        new_values = [item for item in closest_clusts for _ in range(dymaic_length)]
        new_values = new_values[:total_questions]
        cluster[key] = new_values
        list_i = list(train_students_split[ind])
        list_i.append(cluster[key])
        train_students_split[ind] = tuple(list_i)

    print("Starting cluster assignment for validation set...")
    for ind, i in enumerate(val_cluster_data):
        inst = torch.Tensor(i[:feature_dims])
        closest_clusts = []
        q_seq = val_students_split[ind][3]
        total_questions = len(q_seq)
        for virtual_std in range(len(kc_set_list_divid_val[ind])):
            select_kc = kc_set_list_divid_val[ind][virtual_std]
            min_dist = float('inf')
            closest_clust = None

            for j in range(num_clust):
                cur_dist = euclideanDistance(inst, centroids[j], select_kc)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = j

            closest_clusts.append(closest_clust)

        key = (int(i[-2]), int(i[-1]))

        new_values = [item for item in closest_clusts for _ in range(dymaic_length)]
        new_values = new_values[:total_questions]
        cluster[key] = new_values

        list_i = list(val_students_split[ind])
        list_i.append(cluster[key])
        val_students_split[ind] = tuple(list_i)

    print("Starting cluster assignment for test set...")
    for ind, i in enumerate(test_cluster_data):
        inst = torch.Tensor(i[:feature_dims])
        closest_clusts = []
        q_seq = test_students_split[ind][3]
        total_questions = len(q_seq)
        for virtual_std in range(len(kc_set_list_divid_test[ind])):
            select_kc = kc_set_list_divid_test[ind][virtual_std]
            min_dist = float('inf')
            closest_clust = None

            for j in range(num_clust):
                cur_dist = euclideanDistance(inst, centroids[j], select_kc)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = j

            closest_clusts.append(closest_clust)

        key = (int(i[-2]), int(i[-1]))

        new_values = [item for item in closest_clusts for _ in range(dymaic_length)]
        new_values = new_values[:total_questions]
        cluster[key] = new_values

        list_i = list(test_students_split[ind])
        list_i.append(cluster[key])
        test_students_split[ind] = tuple(list_i)

    print("Emotion label assignment completed!")
    return cluster


def add_emotion_labels_to_split_seqs(split_seqs_for_cluster, cluster_dict):
    updated_split_seqs = []

    for split_seq in split_seqs_for_cluster:
        key = (split_seq['student_id'], split_seq['subseq_idx'])

        if key in cluster_dict:
            emotion_labels = cluster_dict[key]
            split_seq['emotion_labels'] = emotion_labels
        else:
            print(f"Warning: Emotion labels not found for segment {key}")
            split_seq['emotion_labels'] = []

        updated_split_seqs.append(split_seq)

    return updated_split_seqs