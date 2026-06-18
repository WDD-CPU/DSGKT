from Gate import WeightedLearningGate, WeightedForgetGate, CJDLoss
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch

def calculate_comprehensive_skill_weights(
        sequences: List[dict], q_matrix: torch.Tensor, num_skills: int,
        alpha: float = 0.5
) -> torch.Tensor:
    """
    计算知识点的综合权重（频率 + 难度），比例均衡且可解释。

    权重 = alpha * 归一化频率 + (1 - alpha) * 难度
    - 频率：知识点被练习的次数（反映覆盖广度）
    - 难度：1 - 正确率（反映掌握难度，使用 Laplace 平滑）

    Args:
        sequences: 学生答题序列列表，每个 dict 应包含 'skill_seq' 和 'response_seq'
        num_skills: 知识点总数
        alpha: 频率权重占比（0～1）。alpha=1 → 只看频率；alpha=0 → 只看难度

    Returns:
        若传入num_skills=5（共 5 个知识点），则函数返回值为形状[5]的一维张量，例如：tensor([0.18, 0.22, 0.20, 0.19, 0.21], dtype=torch.float32)。
        torch.Tensor: 形状为 [num_skills] 的归一化权重向量，元素 ∈ [0,1]，总和为 1
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")

    freq = np.zeros(num_skills)
    correct = np.zeros(num_skills)
    total = np.zeros(num_skills)

    # 遍历所有学生序列，统计频次与正确数
    for seq in sequences:
        if 'skill_seq' not in seq or 'response_seq' not in seq:
            continue
        skill_seq = seq['skill_seq']
        response_seq = seq['response_seq']
        for skill, resp in zip(skill_seq, response_seq):
            if isinstance(skill, (int, np.integer)) and 0 <= skill < num_skills:
                freq[skill] += 1
                total[skill] += 1
                if resp == 1:
                    correct[skill] += 1

    # === 难度计算（带平滑）===
    # Laplace 平滑避免除零：(correct + 1) / (total + 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = (correct + 1) / (total + 2)
    difficulty = 1.0 - accuracy  # 难度 ∈ [0, 1]

    # === 频率归一化 ===
    if freq.sum() > 0:
        norm_freq = freq / freq.sum()  # 归一化到概率分布
    else:
        norm_freq = np.ones(num_skills) / num_skills

    # === 综合权重 ===
    weights = alpha * norm_freq + (1 - alpha) * difficulty

    # 最终归一化（确保 sum=1，数值稳定）
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(num_skills) / num_skills

    return torch.tensor(weights, dtype=torch.float32)



class DSGEKT(nn.Module):
    def __init__(self,num_exercises: int, num_skills: int, sequences=None,
                 q_matrix=None, hidden_dim: int = 128, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        self.num_exercises = num_exercises
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.max_short_window = 10

        if sequences is not None:
            skill_weights_value = calculate_comprehensive_skill_weights(
                sequences, q_matrix, num_skills
            )
        else:
            print("Warning: No sequences provided for skill weight calculation.")
            skill_weights_value = torch.ones(num_skills) / num_skills

        self.register_buffer("skill_weights", skill_weights_value)

        self.initial_knowledge = nn.Parameter(torch.zeros(1, hidden_dim))

        # 离散特征嵌入
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.skill_embed = nn.Embedding(num_skills, embed_dim)
        self.response_embed = nn.Embedding(2, embed_dim)
        self.emotion_embed = nn.Embedding(4, embed_dim)

        nn.init.xavier_uniform_(self.exercise_embed.weight)
        nn.init.xavier_uniform_(self.skill_embed.weight)
        nn.init.xavier_uniform_(self.response_embed.weight)
        nn.init.xavier_uniform_(self.emotion_embed.weight)

        # 长期状态更新门
        self.learning_gate = WeightedLearningGate(hidden_dim, embed_dim)
        self.forget_gate = WeightedForgetGate(hidden_dim, embed_dim)

        # 连续特征投影
        self.proj_time = nn.Sequential(nn.Linear(1, embed_dim))
        self.proj_interval = nn.Sequential(nn.Linear(1, embed_dim))
        self.proj_attempt = nn.Sequential(nn.Linear(1, embed_dim))
        self.proj_hint = nn.Sequential(nn.Linear(1, embed_dim))

        # 特征融合
        self.interaction_fusion = nn.Sequential(nn.Linear(embed_dim * 4, hidden_dim), nn.ReLU())
        self.behavior_fusion = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU())
        self.time_fusion = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU())

        self.features_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU())

        # 短期 Transformer
        self.short_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=dropout,
                activation="relu"
            ),
            num_layers=1
        )

        # 情感 GAT
        self.feat_proj = nn.ModuleList([self._build_proj() for _ in range(4)])
        self.emo_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 1),
            # nn.ReLU(),
            # nn.Linear(embed_dim, 1)
        )
        self.gat = GATv2Conv(in_channels=embed_dim, out_channels=embed_dim, heads=1, concat=False, edge_dim=1)

        # 主预测头：融合状态预测
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 蒸馏用预测头：长期状态预测
        self.predictor_long = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 蒸馏用预测头：短期状态预测
        self.predictor_short = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.norm_fusion = nn.LayerNorm(hidden_dim)

    def _build_proj(self):
        return nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ReLU()
        )

    def compute_interaction_embedding(self, ex_emb, sk_emb, res_emb, emotional_modual):
        interaction_all = torch.cat(
            [ex_emb, sk_emb, res_emb, emotional_modual],
            dim=-1
        )
        return self.interaction_fusion(interaction_all)

    def compute_behavior_embedding(self, attempt_emb, hint_emb):
        behavior_all = torch.cat([attempt_emb, hint_emb], dim=-1)
        return self.behavior_fusion(behavior_all)

    def calculate_time_embedding(self, time_emb, interval_emb):
        time_all = torch.cat([time_emb, interval_emb], dim=-1)
        return self.time_fusion(time_all)

    def fusion_features(self, inter_emb, beh_emb, time_emb):
        interaction_all = torch.cat(
            [inter_emb, beh_emb, time_emb],
            dim=-1
        )
        return self.features_fusion(interaction_all)



    def emotional_modulation(self, time_raw, interval_raw, attempt_raw, hint_raw, emo_emb):
        feats = [time_raw, interval_raw, attempt_raw, hint_raw]

        time_emb, interval_emb, attempt_emb, hint_emb = [
            proj(x) for proj, x in zip(self.feat_proj, feats)
        ]

        emo_proj = self.emo_proj(emo_emb)

        batch_graph = self._build_batch_sparse_graph(
            time_emb,
            interval_emb,
            attempt_emb,
            hint_emb,
            emo_proj
        )

        # 如果你的 GAT 支持 edge_attr / edge_weight
        x_gnn = self.gat(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr
        )

        emo_indices = torch.arange(
            4,
            batch_graph.num_nodes,
            5,
            device=time_raw.device
        )

        emo_gnn = x_gnn[emo_indices].view(*emo_emb.shape)

        return emo_gnn + emo_emb

    def _build_batch_sparse_graph(self, time_emb, interval_emb, attempt_emb, hint_emb, emo_emb):
        B, L, D = time_emb.shape
        N = B * L
        device = time_emb.device

        # [N, 4, D]
        feat_nodes = torch.stack(
            [time_emb, interval_emb, attempt_emb, hint_emb],
            dim=-2
        ).reshape(N, 4, D)

        # [N, D]
        emo_nodes = emo_emb.reshape(N, D)

        # [N]
        base = torch.arange(N, device=device) * 5

        # 行为节点索引: 每个小图前4个节点
        # [N, 4]
        feat_idx = base.view(-1, 1) + torch.arange(4, device=device)

        # 情感节点索引: 每个小图第5个节点
        # [N, 4]
        emo_idx = (base + 4).view(-1, 1).expand(-1, 4)

        # ===== 可学习动态边权 =====
        # [N, 4, D]
        emo_expand = emo_nodes.unsqueeze(1).expand(-1, 4, -1)

        # [N, 4, 3D]
        edge_input = torch.cat(
            [
                feat_nodes,
                emo_expand,
                feat_nodes * emo_expand
            ],
            dim=-1
        )

        # [N, 4]
        edge_weight = torch.sigmoid(
            self.edge_mlp(edge_input).squeeze(-1)
        )

        # feat -> emo
        src = feat_idx.reshape(-1)
        dst = emo_idx.reshape(-1)

        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = edge_weight.reshape(-1, 1)

        # 所有节点: 每个时间步 4个行为节点 + 1个情感节点
        # [N, 5, D] -> [N*5, D]
        all_nodes = torch.cat(
            [
                feat_nodes,
                emo_nodes.unsqueeze(1)
            ],
            dim=1
        ).reshape(N * 5, D)

        batch_vec = torch.arange(N, device=device).repeat_interleave(5)

        return Batch(
            x=all_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_vec,
            num_graphs=N
        )

    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, attempt_seq, hint_seq, emotion_labels,
                q_matrix,
                learn_weights=None,
                forget_weights=None,
                compute_Dloss: bool = False):

        B, L = exercise_seq.shape
        device = exercise_seq.device

        # ===================== 1. 离散特征嵌入 =====================
        ex_emb = self.exercise_embed(exercise_seq.long())
        sk_emb = self.skill_embed(skill_seq.long())
        res_emb = self.response_embed(response_seq.long())
        emo_emb = self.emotion_embed(emotion_labels.long())

        # ===================== 2. 连续特征投影 =====================
        time_raw = time_seq.float().unsqueeze(-1)
        interval_raw = interval_seq.float().unsqueeze(-1)
        attempt_raw = attempt_seq.float().unsqueeze(-1)
        hint_raw = hint_seq.float().unsqueeze(-1)

        time_emb = self.proj_time(time_raw)
        interval_emb = self.proj_interval(interval_raw)
        attempt_emb = self.proj_attempt(attempt_raw)
        hint_emb = self.proj_hint(hint_raw)

        # ===================== 3. 情感增强 =====================
        emotional_modual = self.emotional_modulation(
            time_raw,
            interval_raw,
            attempt_raw,
            hint_raw,
            emo_emb
        )

        # ===================== 4. 多层特征编码 =====================
        curr_interaction = self.compute_interaction_embedding(
            ex_emb,
            sk_emb,
            res_emb,
            emotional_modual
            # emo_emb
        )

        curr_behavior = self.compute_behavior_embedding(
            attempt_emb,
            hint_emb
        )

        curr_time = self.calculate_time_embedding(
            time_emb,
            interval_emb
        )

        fused_features = self.fusion_features(
            curr_interaction,
            curr_behavior,
            curr_time
        )

        # ===================== 5. 状态初始化 =====================
        h_long = self.initial_knowledge.expand(B, self.hidden_dim)

        short_memory = torch.zeros(
            B,
            self.max_short_window,
            self.hidden_dim,
            device=device
        )

        pred_list = []
        distill_list = []

        # ===================== 6. 时序循环 =====================
        for t in range(L):
            _, win_len, _ = short_memory.shape

            # ---------- 短期状态编码 ----------
            short_in = short_memory

            mask = torch.triu(
                torch.ones(win_len, win_len, device=device),
                diagonal=1
            )
            mask = mask.masked_fill(mask == 1, float("-inf"))

            short_encoded = self.short_transformer(short_in, mask=mask)
            h_short = short_encoded[:, -1]

            # ---------- 长短期融合 ----------
            # alpha = self.fusion_knowledge_data(
            #     torch.cat([h_long, h_short], dim=-1)
            # )
            # h_final = alpha * h_long + (1.0 - alpha) * h_short
            h_final = h_long + h_short
            h_final = self.norm_fusion(h_final)

            # ---------- 主预测 ----------
            pred_input = torch.cat(
                [h_final, ex_emb[:, t], sk_emb[:, t]],
                dim=-1
            )
            # pred_input = torch.cat(
            #     [h_short, ex_emb[:, t], sk_emb[:, t]],
            #     dim=-1
            # )

            pred = self.predictor(pred_input).squeeze(-1)
            pred_list.append(pred)

            # ---------- 预测层蒸馏 ----------
            if compute_Dloss:
                pred_long_input = torch.cat(
                    [h_long, ex_emb[:, t], sk_emb[:, t]],
                    dim=-1
                )

                pred_short_input = torch.cat(
                    [h_short, ex_emb[:, t], sk_emb[:, t]],
                    dim=-1
                )

                pred_long = self.predictor_long(pred_long_input).squeeze(-1)
                pred_short = self.predictor_short(pred_short_input).squeeze(-1)

                distill_loss_t = F.mse_loss(
                    pred_short,
                    pred_long.detach()
                )
                distill_list.append(distill_loss_t)

            # ---------- 长期状态更新 ----------
            ff_t = fused_features[:, t]
            # ff_t = curr_interaction[:, t]

            forget_gate = self.forget_gate(h_long, ff_t)
            learning_gate, candidate = self.learning_gate(h_long, ff_t)

            retain_weight = forget_gate
            learn_weight = (1.0 - forget_gate) * learning_gate

            h_long = retain_weight * h_long + learn_weight * candidate

            # ---------- 短期记忆更新 ----------
            short_memory = torch.cat(
                [short_memory[:, 1:], ff_t.unsqueeze(1)],
                dim=1
            )

        pred_seq = torch.stack(pred_list, dim=1)

        if compute_Dloss:
            distill_loss = torch.stack(distill_list).mean()
            return pred_seq, distill_loss
        else:
            return pred_seq, None

    def loss(self,
             pred_seq,
             response_seq,
             mask_seq,
             D_loss=None,
             is_train: bool = True):
        """
        总损失 = KT预测损失 + 预测层蒸馏损失
        D_loss: long prediction 与 short prediction 的 MSE 蒸馏损失
        """

        batch_size, seq_len = mask_seq.shape

        indices = torch.arange(seq_len, device=mask_seq.device).unsqueeze(0)
        indices = indices.expand(batch_size, seq_len)

        exclude_first = indices != 0

        base_mask = mask_seq.bool()
        final_mask = base_mask & exclude_first

        if final_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_seq.device)

        valid_preds = pred_seq[final_mask]
        valid_targets = response_seq[final_mask]

        if torch.isnan(valid_preds).any() or torch.isnan(valid_targets).any():
            print(
                f"警告: 发现NaN值, "
                f"preds: {torch.isnan(valid_preds).sum()}, "
                f"targets: {torch.isnan(valid_targets).sum()}"
            )
            return torch.tensor(0.0, device=pred_seq.device)

        pred_loss = F.binary_cross_entropy(
            valid_preds,
            valid_targets.float()
        )

        if D_loss is not None and is_train:
            total_loss = pred_loss + 0.05 * D_loss
        else:
            total_loss = pred_loss

        return total_loss