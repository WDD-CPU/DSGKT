import torch
import torch.nn.functional as F
import numpy as np
from data import process_data
import copy
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, mean_squared_error
from data import prepare_batch
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from myGKT import DSGEKT
import matplotlib.pyplot as plt
from Emotional_clustering import emotional_vector_calculation
from Emotional_clustering import is_zero_set, euclideanDistance, dymaic_cluster_std, k_means_clust, add_emotion_labels_to_split_seqs
import sys
import torch.nn as nn
import json
import csv
import os
import pandas as pd
import os
from utils import calculate_student_performance, get_performance_key, classify_students, get_gate_weights, dataset_split, computational_ability_level
import random
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_skill_clusters(cluster_path=".\knowledge_relationship_graph.json"):
    try:
        with open(cluster_path, 'r', encoding='utf-8') as f:
            skill_clusters = json.load(f)
        skill_clusters = {int(k): [int(x) for x in v] for k, v in skill_clusters.items()}
        print(f"Successfully loaded skill relation clusters, total {len(skill_clusters)} core skills")
        return skill_clusters
    except Exception as e:
        print(f"Failed to load relation clusters: {e}, regularization loss will be disabled")
        return {}


def save_sequences_to_expanded_csv(sequences, save_path):
    if not sequences:
        print(f"Data empty, skip saving: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        'student_id',
        'subseq_idx',
        'total_subseqs',
        'original_seq_length',
        'subseq_start_idx',
        'subseq_end_idx',
        'subseq_length',
        'is_last_subseq',
        'exercise_id',
        'skill_id',
        'response',
        'time',
        'interval',
        'hint',
        'attempt',
        'start_time',
        'mask',
        'emotion_label'
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

    print(f"Saved successfully: {save_path}")


def load_expanded_csv_to_sequences(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path} → return empty sequences")
        return []

    df = pd.read_csv(csv_path, encoding='utf-8')
    sequences = []

    groups = df.groupby(['student_id', 'subseq_idx'])

    for (stu_id, sub_idx), group in groups:
        seq = {
            'student_id': int(stu_id),
            'subseq_idx': int(sub_idx),
            'total_subseqs': int(group['total_subseqs'].iloc[0]),
            'original_seq_length': int(group['original_seq_length'].iloc[0]),
            'subseq_start_idx': int(group['subseq_start_idx'].iloc[0]),
            'subseq_end_idx': int(group['subseq_end_idx'].iloc[0]),
            'subseq_length': int(group['subseq_length'].iloc[0]),
            'is_last_subseq': bool(group['is_last_subseq'].iloc[0]),
            'exercise_seq': group['exercise_id'].tolist(),
            'skill_seq': group['skill_id'].tolist(),
            'response_seq': group['response'].tolist(),
            'time_seq': group['time'].tolist(),
            'interval_seq': group['interval'].tolist(),
            'hint_seq': group['hint'].tolist(),
            'attempt_seq': group['attempt'].tolist(),
            'start_time_seq': group['start_time'].tolist(),
            'mask_seq': group['mask'].tolist(),
            'emotion_labels': group['emotion_label'].tolist(),
        }
        sequences.append(seq)

    print(f"Load completed: {csv_path} → total {len(sequences)} sequences")
    return sequences


def train_model_variant(epochs, model_class, model_name, sequences, q_matrix, device, max_seq_len, batch_size, skill_num, **kwargs):
    print(f"\nTraining {model_name}...")

    metrics_filename = f"{model_name}_training_metrics.txt"
    print(f"Training metrics will be saved to: {metrics_filename}")
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training params: epochs={epochs}, batch_size={batch_size}\n")
        f.write("=" * 80 + "\n\n")
        f.write("epoch\ttrain_loss\tval_loss\tval_auc\tval_acc\tval_rmse\tval_r2\ttest_loss\ttest_auc\ttest_acc\ttest_rmse\ttest_r2\n")

    print("\nDirectly load preprocessed CSV with emotion labels, skip all preprocessing!")

    train_sequences = load_expanded_csv_to_sequences("./with_emotion_data_2017/train_expanded.csv")
    val_sequences = load_expanded_csv_to_sequences("./with_emotion_data_2017/val_expanded.csv")
    test_sequences = load_expanded_csv_to_sequences("./with_emotion_data_2017/test_expanded.csv")

    train_student_categories, val_student_categories, test_student_categories, student_categories = computational_ability_level(train_sequences, val_sequences, test_sequences)

    model = model_class(sequences=train_sequences, q_matrix=q_matrix, **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=0,
        verbose=True,
        min_lr=1e-6
    )

    best_val_metrics = {'loss': float('inf')}
    best_model = None
    weight_stats = {'epoch': [], 'good_count': [], 'medium_count': [], 'poor_count': []}
    test_metrics_history = []
    lambda_reg = 0.1

    from tqdm import tqdm, trange
    print("\nStarting main training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        processed_samples = 0
        epoch_weight_stats = {'good': 0, 'medium': 0, 'poor': 0}
        debug_batch_idx = np.random.randint(0, len(train_sequences) // batch_size)
        num_batches = 0
        total_batches = (len(train_sequences) + batch_size - 1) // batch_size
        pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch + 1:03d}/{epochs}",
            unit="batch",
            file=sys.stdout,
            leave=True,
            dynamic_ncols=True
        )

        for start_idx in range(0, len(train_sequences), batch_size):
            end_idx = min(start_idx + batch_size, len(train_sequences))
            current_batch = train_sequences[start_idx:end_idx]
            current_batch_size = len(current_batch)

            learn_weights = []
            forget_weights = []
            batch_categories = []
            for seq in current_batch:
                performance_key = get_performance_key(seq)
                category = student_categories.get(performance_key, 'medium')
                batch_categories.append(category)
                epoch_weight_stats[category] += 1
                l_w, f_w = get_gate_weights(category)
                learn_weights.append(l_w)
                forget_weights.append(f_w)

            learn_weights = torch.tensor(learn_weights, device=device).float()
            forget_weights = torch.tensor(forget_weights, device=device).float()
            batch_num = start_idx // batch_size
            batch = prepare_batch(current_batch, current_batch_size, device)

            pred_seq, CJDLoss = model(
                exercise_seq=batch['exercise_seq'],
                skill_seq=batch['skill_seq'],
                response_seq=batch['response_seq'],
                time_seq=batch['time_seq'],
                interval_seq=batch['interval_seq'],
                attempt_seq=batch['attempt_seq'],
                hint_seq=batch['hint_seq'],
                emotion_labels=batch['emotion_labels'],
                q_matrix=q_matrix,
                learn_weights=None,
                forget_weights=None,
                compute_Dloss=True
            )

            loss_kt = model.loss(pred_seq, batch['response_seq'], batch['mask_seq'], CJDLoss, is_train=True)
            loss = loss_kt

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            processed_samples += current_batch_size

            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                'Epoch': f"{epoch + 1}/{epochs}",
                'Batch': f"{num_batches}/{(len(train_sequences) - 1) // batch_size + 1}",
                'AvgTrainLoss': f"{avg_loss:.4f}",
                'GradNorm': f"{grad_norm:.2f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            pbar.update(1)
        pbar.close()

        weight_stats['epoch'].append(epoch + 1)
        weight_stats['good_count'].append(epoch_weight_stats['good'])
        weight_stats['medium_count'].append(epoch_weight_stats['medium'])
        weight_stats['poor_count'].append(epoch_weight_stats['poor'])

        train_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_model(model, val_sequences, student_categories, q_matrix, device)
        test_metrics = evaluate_model(model, test_sequences, student_categories, q_matrix, device)
        test_metrics_history.append(test_metrics)

        with open(metrics_filename, 'a', encoding='utf-8') as f:
            f.write(
                f"{epoch + 1}\t{train_loss:.6f}\t{val_metrics['loss']:.6f}\t{val_metrics['AUC']:.6f}\t{val_metrics['ACC']:.6f}\t{val_metrics['RMSE']:.6f}\t{val_metrics['R2']:.6f}\t")
            f.write(
                f"{test_metrics['loss']:.6f}\t{test_metrics['AUC']:.6f}\t{test_metrics['ACC']:.6f}\t{test_metrics['RMSE']:.6f}\t{test_metrics['R2']:.6f}\n")

        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            best_model = copy.deepcopy(model)

        print("\t", f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val AUC: {val_metrics['AUC']:.4f}, Val ACC: {val_metrics['ACC']:.4f}, Val RMSE: {val_metrics['RMSE']:.4f}")
        print("\t", f"Test Loss: {test_metrics['loss']:.4f}, Test AUC: {test_metrics['AUC']:.4f}, "
              f"Test ACC: {test_metrics['ACC']:.4f}, Test RMSE: {test_metrics['RMSE']:.4f}")

        scheduler.step(val_metrics['loss'])

    print("\nWeight usage statistics during training:")
    print("Epoch\tGood\tMedium\tPoor")
    for i in range(len(weight_stats['epoch'])):
        print(
            f"{weight_stats['epoch'][i]}\t{weight_stats['good_count'][i]}\t{weight_stats['medium_count'][i]}\t{weight_stats['poor_count'][i]}")

    test_metrics = evaluate_model(best_model, test_sequences, student_categories, q_matrix, device)
    print("\n" + "=" * 60)
    print(f"{model_name} - Final performance of best model on test set:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   AUC : {test_metrics['AUC']:.4f}")
    print(f"   ACC : {test_metrics['ACC']:.4f}")
    print(f"   RMSE: {test_metrics['RMSE']:.4f}")
    print(f"   F1  : {test_metrics.get('F1', 0.0):.4f}")
    print(f"   R2  : {test_metrics.get('R2', 0.0):.4f}")
    print("=" * 60 + "\n")
    return best_model, test_metrics


def evaluate_model(model, sequences, student_categories, q_matrix, device, batch_size=256):
    model.eval()
    total_loss = 0
    num_valid_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            current_batch = sequences[i:i + batch_size]
            batch = prepare_batch(current_batch, batch_size, device)

            learn_weights = []
            forget_weights = []
            for seq in current_batch:
                performance_key = get_performance_key(seq)
                category = student_categories.get(performance_key, 'medium')
                l_w, f_w = get_gate_weights(category)
                learn_weights.append(l_w)
                forget_weights.append(f_w)

            learn_weights = torch.tensor(learn_weights, device=device).float()
            forget_weights = torch.tensor(forget_weights, device=device).float()

            pred_seq, _ = model(
                exercise_seq=batch['exercise_seq'],
                skill_seq=batch['skill_seq'],
                response_seq=batch['response_seq'],
                time_seq=batch['time_seq'],
                interval_seq=batch['interval_seq'],
                attempt_seq=batch['attempt_seq'],
                hint_seq=batch['hint_seq'],
                emotion_labels=batch['emotion_labels'],
                q_matrix=q_matrix,
                learn_weights=None,
                forget_weights=None,
                compute_Dloss=False
            )

            batch_size_, seq_len = batch['mask_seq'].shape
            indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size_, seq_len)
            exclude_first = indices != 0
            base_mask = batch['mask_seq'].bool()
            eval_mask = base_mask & exclude_first

            if eval_mask.sum() == 0:
                continue

            valid_preds = pred_seq[eval_mask]
            valid_targets = batch['response_seq'][eval_mask]

            all_preds.extend(valid_preds.cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())

            loss = model.loss(pred_seq, batch['response_seq'], batch['mask_seq'], D_loss=None, is_train=False)
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_valid_batches += 1

    if not all_preds or not all_targets:
        return {'loss': float('inf'), 'AUC': 0.0, 'ACC': 0.0, 'RMSE': float('inf'), 'R2': 0.0, 'F1': 0.0}

    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    if num_valid_batches > 0:
        metrics['loss'] = total_loss / num_valid_batches
    else:
        metrics['loss'] = float('inf')
    return metrics


def calculate_metrics(y_true, y_pred, threshold=0.5):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    auc = roc_auc_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    logloss = log_loss(y_true, y_pred_clipped)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

    y_pred_binary = (y_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    return {
        'AUC': auc,
        'ACC': acc,
        'RMSE': rmse,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'LogLoss': logloss,
        'R2': r2,
        'threshold': threshold
    }


def set_random_seed(seed=2027):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main(data_path, max_seq_len, use_cuda, random_seed, epoch, batch_size, skill_num):
    set_random_seed(random_seed)
    sequences, q_matrix, num_exercises, num_skills = process_data(
        data_path,
        max_seq_len=max_seq_len
    )
    q_matrix = torch.tensor(q_matrix, dtype=torch.float)

    if use_cuda == False:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_matrix = q_matrix.to(device)

    model_params = {
        'num_exercises': num_exercises,
        'num_skills': num_skills,
        'hidden_dim': 128,
        'embed_dim': 128
    }

    variants = [
        (DSGEKT, 'weighted_model'),
    ]

    results = {}
    for model_class, model_name in variants:
        model, test_metrics = train_model_variant(
            epoch,
            model_class,
            model_name,
            sequences,
            q_matrix,
            device,
            max_seq_len,
            batch_size,
            skill_num,
            **model_params
        )
        results[model_name] = test_metrics
        torch.save(model.state_dict(), f'{model_name}.pth')


if __name__ == "__main__":
    data_path = 'F:./processed_data_2017/processed_data17.csv'
    max_seq_len = 200
    use_cuda = True
    random_seed = 2027
    epoch = 50
    batch_size = 64
    skill_num = 102
    main(data_path, max_seq_len, use_cuda, random_seed, epoch, batch_size, skill_num)