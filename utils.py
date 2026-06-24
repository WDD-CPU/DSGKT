import torch
import numpy as np
import pandas as pd


def calculate_student_performance(sequences):
    """Calculate local proficiency based on segmented student response sequences

    Args:
        sequences: segmented student response sequence data

    Returns:
        dict:{ (student_id, subseq_idx) : accuracy,
               (student_id, subseq_idx) : accuracy
               }
    """
    student_performance = {}

    for seq in sequences:
        student_id = seq['student_id']
        subseq_idx = seq.get('subseq_idx', 0)

        performance_key = (student_id, subseq_idx)

        responses = seq['response_seq']
        valid_responses = [r for r in responses if r != -1]
        if valid_responses:
            accuracy = sum(valid_responses) / len(valid_responses)
            student_performance[performance_key] = accuracy

    return student_performance


def get_performance_key(seq):
    student_id = seq['student_id']
    subseq_idx = seq.get('subseq_idx', 0)
    return (student_id, subseq_idx)


def classify_students(student_performance):
    """Classify students by accuracy of segmented subsequences

    Args:
        student_performance: mapping from (student_id, subseq_idx) tuple to accuracy value

    Returns:
        dict: mapping from (student_id, subseq_idx) tuple to category label
    """
    student_categories = {}

    for performance_key, accuracy in student_performance.items():
        if accuracy > 0.518:
            student_categories[performance_key] = 'good'
        elif accuracy > 0.280:
            student_categories[performance_key] = 'medium'
        else:
            student_categories[performance_key] = 'poor'

    return student_categories


def get_gate_weights(student_category):
    weights = {
        'good': (1.15, 1.0),
        'medium': (1.0, 1.0),
        'poor': (0.85, 0.85)
    }
    return weights.get(student_category, (1.0, 1.0))


def computational_ability_level(train_sequences, val_sequences, test_sequences):
    print("\n"+'❌' * 70)
    print("==========================Start calculating student proficiency levels based on local subsequence performance...==========================")
    train_student_performance = calculate_student_performance(train_sequences)
    train_student_categories = classify_students(train_student_performance)
    val_student_performance = calculate_student_performance(val_sequences)
    val_student_categories = classify_students(val_student_performance)
    test_student_performance = calculate_student_performance(test_sequences)
    test_student_categories = classify_students(test_student_performance)
    student_categories = {}
    student_categories.update(train_student_categories)
    student_categories.update(val_student_categories)
    student_categories.update(test_student_categories)
    print("==========================Student proficiency level calculation finished...==========================")
    print('❌' * 70)
    return train_student_categories, val_student_categories, test_student_categories, student_categories


def dataset_split(sequences):
    print('\n')
    print('✅' * 70)
    print("========================Start dataset splitting========================")

    indices = np.random.permutation(len(sequences))

    test_size = 0
    train_val_size = len(sequences) - test_size

    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size

    test_sequences = []
    train_sequences = [sequences[i] for i in indices[:train_size]]
    val_sequences = [sequences[i] for i in indices[train_size:train_size+val_size]]

    print('Dataset split result (Train:80% | Val:20% | Test: empty)')
    print(f'Train set: {len(train_sequences)} records')
    print(f'Validation set: {len(val_sequences)} records')
    print(f'Test set: {len(test_sequences)} records')
    print(f'Total combined: {len(train_sequences)+len(val_sequences)+len(test_sequences)} records, expected: {len(sequences)}')

    print("\n\033[32mDataset splitting completed ✅\033[0m")
    print('✅' * 70)
    return train_sequences, val_sequences, test_sequences


def calculate_skill_weights(training_data_path, num_skills):
    try:
        df = pd.read_csv(training_data_path)
    except FileNotFoundError:
        df = pd.read_csv('student_log_12.csv')
        if 'skill_id' not in df.columns and 'skill' in df.columns:
            print("Creating skill_id from skill column")
            skill_map = {skill: idx for idx, skill in enumerate(sorted(df['skill'].unique()))}
            df['skill_id'] = df['skill'].map(skill_map)
    except Exception as e:
        print(f"Warning: Could not calculate skill weights: {e}")
        return torch.ones(num_skills)

    if 'skill_id' not in df.columns:
        print("Warning: Could not calculate skill weights: 'skill_id' column not found")
        return torch.ones(num_skills)

    df['skill_id'] = df['skill_id'].astype(int)

    skill_counts = df['skill_id'].value_counts()
    total_count = skill_counts.sum()

    skill_weights = skill_counts / total_count
    weights_tensor = torch.ones(num_skills)

    for skill_id, weight in skill_weights.items():
        if 0 <= skill_id < num_skills:
            weights_tensor[skill_id] = weight

    print(f"Skill weights calculated for {len(skill_weights)} skills")
    print(f"Weights range: [{weights_tensor.min().item():.4f}, {weights_tensor.max().item():.4f}]")

    return weights_tensor