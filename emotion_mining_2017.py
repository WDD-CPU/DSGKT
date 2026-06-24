import torch
import numpy as np
from data import process_data
import matplotlib.pyplot as plt
from Emotional_clustering import emotional_vector_calculation
from Emotional_clustering import k_means_clust, add_emotion_labels_to_split_seqs
import csv
import os
from utils import dataset_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import random


def set_random_seed(seed=2027):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_sequences_to_expanded_csv(sequences, save_path):
    if not sequences:
        print(f"⚠️ Data is empty, skipping save: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        'student_id', 'subseq_idx', 'total_subseqs', 'original_seq_length',
        'subseq_start_idx', 'subseq_end_idx', 'subseq_length', 'is_last_subseq',
        'exercise_id', 'skill_id', 'response', 'time', 'interval', 'hint', 'attempt',
        'start_time', 'mask', 'emotion_label'
    ]

    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seq in sequences:
            student_id = seq['student_id']
            subseq_idx = seq['subseq_idx']
            total_subseqs = seq['total_subseqs']
            original_seq_length = seq['original_seq_length']
            subseq_start_idx = seq['subseq_start_idx']
            subseq_end_idx = seq['subseq_end_idx']
            subseq_length = seq['subseq_length']
            is_last_subseq = seq['is_last_subseq']

            exercise_seq = seq.get('exercise_seq', [])
            skill_seq = seq.get('skill_seq', [])
            response_seq = seq.get('response_seq', [])
            time_seq = seq.get('time_seq', [])
            interval_seq = seq.get('interval_seq', [])
            hint_seq = seq.get('hint_seq', [])
            attempt_seq = seq.get('attempt_seq', [])
            start_time_seq = seq.get('start_time_seq', [])
            mask_seq = seq.get('mask_seq', [])
            emotion_labels = seq.get('emotion_labels', [])

            seq_len = len(exercise_seq)
            for i in range(seq_len):
                row = {
                    'student_id': student_id,
                    'subseq_idx': subseq_idx,
                    'total_subseqs': total_subseqs,
                    'original_seq_length': original_seq_length,
                    'subseq_start_idx': subseq_start_idx,
                    'subseq_end_idx': subseq_end_idx,
                    'subseq_length': subseq_length,
                    'is_last_subseq': int(is_last_subseq),

                    'exercise_id': exercise_seq[i],
                    'skill_id': skill_seq[i] if i < len(skill_seq) else '',
                    'response': response_seq[i] if i < len(response_seq) else '',
                    'time': time_seq[i] if i < len(time_seq) else '',
                    'interval': interval_seq[i] if i < len(interval_seq) else '',
                    'hint': hint_seq[i] if i < len(hint_seq) else '',
                    'attempt': attempt_seq[i] if i < len(attempt_seq) else '',
                    'start_time': start_time_seq[i] if i < len(start_time_seq) else '',
                    'mask': mask_seq[i] if i < len(mask_seq) else '',
                    'emotion_label': emotion_labels[i] if i < len(emotion_labels) else ''
                }
                writer.writerow(row)

    print(f"✅ Saved successfully: {save_path}")


if __name__ == "__main__":
    data_path = './processed_data_2017/processed_data17.csv'
    max_seq_len = 200
    use_cuda = True
    random_seed = 2027
    skill_num = 102

    set_random_seed(random_seed)

    print("🔹 Loading raw data...")
    sequences, q_matrix, num_exercises, num_skills = process_data(data_path, max_seq_len=max_seq_len)

    print("🔹 Splitting train/val/test sets...")
    train_sequences, val_sequences, test_sequences = dataset_split(sequences)

    print('\n' + '⬇️' * 70)
    print("========================== Data Splitting & Augmentation ==========================")
    train_students_all, train_students_split, train_cluster_data, train_ids, train_max_skills, train_max_seg, train_max_attempts, train_sequences = emotional_vector_calculation(
        train_sequences, skill_num=skill_num, max_len=max_seq_len)
    val_students_all, val_students_split, val_cluster_data, val_ids, val_max_skills, val_max_seg, val_max_attempts, val_sequences = emotional_vector_calculation(
        val_sequences, skill_num=skill_num, max_len=max_seq_len)
    test_students_all, test_students_split, test_cluster_data, test_ids, test_max_skills, test_max_seg, test_max_attempts, test_sequences = emotional_vector_calculation(
        test_sequences, skill_num=skill_num, max_len=max_seq_len)

    max_skills = max([int(train_max_skills), int(val_max_skills), int(test_max_skills)]) + 1
    max_stu = max(train_ids + test_ids + val_ids) + 1
    max_seg = max([int(train_max_seg), int(val_max_seg), int(test_max_seg)]) + 1
    print(f"Num_skills: {max_skills}, Max_stu: {max_stu}, Max_seg: {max_seg}")
    print('🔥 Split data size: train=%s, val=%s, test=%s' % (len(train_students_split), len(val_students_split), len(test_students_split)))
    print('⬆️' * 70 + '\n')

    print('🔴' * 70)
    print("========================== Emotion Clustering ==========================")
    cluster = k_means_clust(
        train_students_split, val_students_split, test_students_split,
        train_cluster_data, val_cluster_data, test_cluster_data,
        max_stu, max_seg, 4, max_skills
    )

    train_sequences = add_emotion_labels_to_split_seqs(train_sequences, cluster)
    val_sequences = add_emotion_labels_to_split_seqs(val_sequences, cluster)
    test_sequences = add_emotion_labels_to_split_seqs(test_sequences, cluster)
    print("✅ Emotion labels added successfully")
    print('🔴' * 70 + '\n')

    print("💾 Saving expanded CSV with emotion labels...")
    save_root = "./with_emotion_data_2017/"
    save_sequences_to_expanded_csv(train_sequences, os.path.join(save_root, "train_expanded.csv"))
    save_sequences_to_expanded_csv(val_sequences, os.path.join(save_root, "val_expanded.csv"))
    save_sequences_to_expanded_csv(test_sequences, os.path.join(save_root, "test_expanded.csv"))

    print("\n🎉 All done! Files saved to: ", save_root)