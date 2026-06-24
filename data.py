import pandas as pd
import numpy as np
import torch


def process_data(input_file, max_seq_len=200):
    df = pd.read_csv(input_file)
    if (df['response_time'] < 0).any():
        neg_rt_count = (df['response_time'] < 0).sum()
        print(f"⚠️ Warning: Found {neg_rt_count} negative values in original response_time, corrected to 0.")
        df['response_time'] = df['response_time'].clip(lower=0)
    else:
        print("✅ No negative values in original response_time column.")

    neg_count = (df['time_interval'] < 0).sum()
    total_count = len(df)
    neg_ratio = neg_count / total_count if total_count > 0 else 0

    if neg_count > 0:
        print(f"⚠️ Warning: Found {neg_count} negative records in original time_interval ({neg_ratio:.4%})")
        print("Sample negative records (first 5):")
        print(df[df['time_interval'] < 0][['student_id', 'exercise_id', 'start_time', 'time_interval']].head())
    else:
        print("✅ No negative values in original time_interval column, data is normal.")

    df = df.sort_values(['student_id', 'start_time'], ascending=[True, True]).reset_index(drop=True)

    skill_map = {skill: idx for idx, skill in enumerate(sorted(df['skill'].unique()))}
    df['skill_id'] = df['skill'].map(skill_map)

    exercise_map = {ex: idx for idx, ex in enumerate(sorted(df['exercise_id'].unique()))}
    df['exercise_id_mapped'] = df['exercise_id'].map(exercise_map)

    student_map = {student: idx for idx, student in enumerate(sorted(df['student_id'].unique()))}
    df['student_id_mapped'] = df['student_id'].map(student_map)
    num_students = len(student_map)

    import json
    import os

    output_dir = os.path.dirname(input_file)
    mappings_dir = os.path.join(output_dir, 'mappings')
    os.makedirs(mappings_dir, exist_ok=True)

    skill_reverse_map = {int(v): str(k) for k, v in skill_map.items()}
    exercise_reverse_map = {int(v): str(k) for k, v in exercise_map.items()}
    student_reverse_map = {int(v): str(k) for k, v in student_map.items()}

    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj

    mappings_json = {
        'skill_map': convert_to_serializable(skill_map),
        'skill_reverse_map': convert_to_serializable(skill_reverse_map),
        'exercise_map': convert_to_serializable(exercise_map),
        'exercise_reverse_map': convert_to_serializable(exercise_reverse_map),
        'student_map': convert_to_serializable(student_map),
        'student_reverse_map': convert_to_serializable(student_reverse_map)
    }

    json_path = os.path.join(mappings_dir, 'id_mappings.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mappings_json, f, ensure_ascii=False, indent=2)
    print(f"✅ Exercise, skill, and student mapping dictionaries saved (JSON format): {json_path}")

    import pickle
    pkl_path = os.path.join(mappings_dir, 'id_mappings.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'skill_map': skill_map,
            'skill_reverse_map': skill_reverse_map,
            'exercise_map': exercise_map,
            'exercise_reverse_map': exercise_reverse_map,
            'student_map': student_map,
            'student_reverse_map': student_reverse_map
        }, f)
    print(f"✅ Exercise, skill, and student mapping dictionaries saved (PKL format): {pkl_path}")

    num_exercises = len(exercise_map)
    num_skills = len(skill_map)
    q_matrix = np.zeros((num_exercises, num_skills))

    unique_exercise_skills = df[['exercise_id_mapped', 'skill_id']].drop_duplicates()

    exercise_ids = unique_exercise_skills['exercise_id_mapped'].astype(int).values
    skill_ids = unique_exercise_skills['skill_id'].astype(int).values
    q_matrix[exercise_ids, skill_ids] = 1

    sequences = []
    seq_lengths = []

    for student_id_mapped, group in df.groupby('student_id_mapped'):
        group = group.sort_values('start_time')
        seq_length = len(group)
        seq_lengths.append(seq_length)

        original_student_id = group['student_id'].iloc[0]

        seq = {
            'exercise_seq': group['exercise_id_mapped'].values.astype(np.int64),
            'skill_seq': group['skill_id'].values.astype(np.int64),
            'response_seq': np.clip(group['correct'].values, 0, 1).astype(np.float32),
            'time_seq': group['response_time'].values.astype(np.float32),
            'interval_seq': group['time_interval'].values.astype(np.float32),
            'attempt_seq': np.clip(group['attempt_count'].values, 0, None).astype(np.int32),
            'hint_seq': np.clip(group['hint_count'].values, 0, None).astype(np.int32),
            'start_time_seq': group['start_time'].values.astype(np.int64),
            'mask_seq': np.ones(len(group), dtype=np.float32),
            'student_id': student_id_mapped,
            'student_id_original': original_student_id,
            'exercise_original': group['exercise_id'].values,
            'skill_original': group['skill'].values,
        }
        sequences.append(seq)

    seq_data = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq['exercise_seq'])
        for j in range(seq_len):
            seq_data.append({
                'seq_index': i,
                'student_id_mapped': seq['student_id'],
                'student_id_original': seq['student_id_original'],
                'position': j,
                'exercise_id_mapped': seq['exercise_seq'][j],
                'exercise_id_original': seq['exercise_original'][j],
                'skill_id_mapped': seq['skill_seq'][j],
                'skill_original': seq['skill_original'][j],
                'response': seq['response_seq'][j],
                'time': seq['time_seq'][j],
                'interval': seq['interval_seq'][j],
                'attempt': seq['attempt_seq'][j],
                'hint': seq['hint_seq'][j],
                'start_time': seq['start_time_seq'][j],
                'mask': seq['mask_seq'][j]
            })

    seq_df = pd.DataFrame(seq_data)
    seq_csv_path = os.path.join(output_dir, 'processed_sequences.csv')
    seq_df.to_csv(seq_csv_path, index=False)
    print(f"✅ Mapped data saved to: {seq_csv_path}")
    print(f"   Contains {len(sequences)} sequences, {len(seq_df)} records")

    print(f"\nℹℹℹ   Data Preprocessing Statistics:")
    print(f"Original record count: {len(df)}")
    print(f"Number of students (original): {df['student_id'].nunique()}")
    print(f"Number of exercises: {num_exercises}")
    print(f"Number of skills: {num_skills}")
    print(f"Sequence length statistics:")
    print(f"  - Total sequences: {len(seq_lengths)}")
    print(f"  - Min sequence length: {min(seq_lengths)}")
    print(f"  - Max sequence length: {max(seq_lengths)}")
    print(f"  - Mean sequence length: {np.mean(seq_lengths):.2f}")
    print(f"  - Median sequence length: {np.median(seq_lengths):.2f}")
    print(f"  - 👉👉👉 Processed student sequence count: {len(sequences)}")
    print(f"response_time statistics:")
    print(f"  - Range: [{df['response_time'].min():.4f}, {df['response_time'].max():.4f}]")
    print(f"  - Mean: {df['response_time'].mean():.4f}")
    print(f"  - Std: {df['response_time'].std():.4f}")
    print(f"time_interval statistics:")
    print(f"  - Range: [{df['time_interval'].min():.4f}, {df['time_interval'].max():.4f}]")
    print(f"  - Mean: {df['time_interval'].mean():.4f}")
    print(f"  - Std: {df['time_interval'].std():.4f}")
    print(f"Correctness distribution: {df['correct'].value_counts(normalize=True).round(4)}")
    print(f"Q-matrix density: {np.mean(q_matrix):.4f}")
    print(f"Average skills per exercise: {np.mean(np.sum(q_matrix, axis=1)):.2f}")

    return sequences, q_matrix, num_exercises, num_skills


def prepare_batch(sequences, batch_size, device):
    if batch_size == 0:
        raise ValueError("Empty batch")

    max_len = max(len(seq['exercise_seq']) for seq in sequences)

    batch = {
        'exercise_seq': torch.zeros(batch_size, max_len, dtype=torch.long),
        'skill_seq': torch.zeros(batch_size, max_len, dtype=torch.long),
        'response_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'time_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'interval_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'attempt_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'hint_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'start_time_seq': torch.zeros(batch_size, max_len, dtype=torch.long),
        'mask_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'emotion_labels': torch.zeros(batch_size, max_len, dtype=torch.long)
    }

    for i, seq in enumerate(sequences):
        L = len(seq['exercise_seq'])
        if L > 0:
            batch['exercise_seq'][i, :L] = torch.tensor(seq['exercise_seq'], dtype=torch.long)
            batch['skill_seq'][i, :L] = torch.tensor(seq['skill_seq'], dtype=torch.long)
            batch['response_seq'][i, :L] = torch.clamp(torch.tensor(seq['response_seq'], dtype=torch.float), 0, 1)
            batch['time_seq'][i, :L] = torch.tensor(seq['time_seq'], dtype=torch.float)
            batch['interval_seq'][i, :L] = torch.tensor(seq['interval_seq'], dtype=torch.float)
            batch['attempt_seq'][i, :L] = torch.tensor(seq['attempt_seq'], dtype=torch.float)
            batch['hint_seq'][i, :L] = torch.tensor(seq['hint_seq'], dtype=torch.float)
            batch['start_time_seq'][i, :L] = torch.tensor(seq['start_time_seq'], dtype=torch.long)
            batch['mask_seq'][i, :L] = 1.0
            batch['emotion_labels'][i, :L] = torch.tensor(seq['emotion_labels'], dtype=torch.long)

    batch['time_seq'] = torch.log1p(batch['time_seq'])
    batch['interval_seq'] = torch.log1p(batch['interval_seq'])

    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    batch['time_seq'] = normalize(batch['time_seq'])
    batch['interval_seq'] = normalize(batch['interval_seq'])

    return {k: v.to(device) for k, v in batch.items()}