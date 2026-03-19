import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, hist_dim, curr_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hist_dim + curr_dim, hist_dim),
            nn.Sigmoid()
        )
    def forward(self, hist_feat, curr_feat):
        combined = torch.cat([hist_feat, curr_feat], dim=1)
        g = self.gate(combined)
        return torch.cat([hist_feat * g, curr_feat], dim=1)


class AttentionFusion(nn.Module):
    def __init__(self, hist_dims: list, curr_dim: int, d_k=32):
        super(AttentionFusion, self).__init__()
        self.d_k = d_k
        self.k_projs = nn.ModuleList([nn.Linear(dim, d_k) for dim in hist_dims])
        self.v_projs = nn.ModuleList([nn.Linear(dim, d_k) for dim in hist_dims])
        self.q_proj = nn.Linear(curr_dim, d_k)

    def forward(self, hist_feats: list, curr_feat):
        K = torch.stack([proj(hf) for proj, hf in zip(self.k_projs, hist_feats)], dim=1)
        V = torch.stack([proj(hf) for proj, hf in zip(self.v_projs, hist_feats)], dim=1)
        Q = self.q_proj(curr_feat).unsqueeze(1)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V).squeeze(1)

        return torch.cat([context, curr_feat], dim=1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hist_dims: list, curr_dim: int, d_model=32, n_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.d_model = d_model
        self.aligners = nn.ModuleList([nn.Linear(dim, d_model) for dim in hist_dims])
        self.q_proj = nn.Linear(curr_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, hist_feats: list, curr_feat):
        aligned_hist = [aligner(hf).unsqueeze(1) for aligner, hf in zip(self.aligners, hist_feats)]
        memory_bank = torch.cat(aligned_hist, dim=1)

        Q = self.q_proj(curr_feat).unsqueeze(1)
        attn_output, attn_weights = self.cross_attn(query=Q, key=memory_bank, value=memory_bank)

        context_vector = attn_output.squeeze(1)
        return torch.cat([context_vector, curr_feat], dim=1)


class CascadedModel(nn.Module):
    def __init__(self, stage_configs, hidden_dims=None, dropout_rate=0.1, fusion_type='concat'):
        super(CascadedModel, self).__init__()
        self.fusion_type = fusion_type

        if hidden_dims is None:
            hidden_dims = {'s1': [32], 's2': [32], 's3': [32], 's4': [32]}
        self.hd = hidden_dims
        self.dropout_rate = dropout_rate

        self.embeddings = nn.ModuleDict()
        self.stage_dims = {}
        for stage, cfg in stage_configs.items():
            emb_list = nn.ModuleList([nn.Embedding(num_embeddings=dim, embedding_dim=2) for dim in cfg["cat_dims"]])
            self.embeddings[stage] = emb_list
            self.stage_dims[stage] = cfg["num_dim"] + len(cfg["cat_dims"]) * 2

        def build_mlp(in_features, hidden_dims_list, drop):
            layers = []
            curr_in = in_features
            for out_features in hidden_dims_list:
                layers.append(nn.Linear(curr_in, out_features))
                layers.append(nn.LayerNorm(out_features))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(drop))
                curr_in = out_features
            return nn.Sequential(*layers)

        self.s1_layers = build_mlp(self.stage_dims['stage1'], self.hd['s1'], self.dropout_rate)
        self.out_19 = nn.Linear(self.hd['s1'][-1], 1)

        dim_state1 = self.hd['s1'][-1] + 1
        dim_state2 = self.hd['s2'][-1] + 2
        dim_state3 = self.hd['s3'][-1] + 1

        if fusion_type == 'concat':
            s2_input_dim = dim_state1 + self.stage_dims['stage2']
            s3_input_dim = dim_state2 + self.stage_dims['stage3']
            s4_input_dim = dim_state3 + self.hd['s1'][-1] + self.stage_dims['stage4']

        elif fusion_type == 'glu':
            self.glu2 = GatedFusion(hist_dim=dim_state1, curr_dim=self.stage_dims['stage2'])
            s2_input_dim = dim_state1 + self.stage_dims['stage2']
            self.glu3 = GatedFusion(hist_dim=dim_state2, curr_dim=self.stage_dims['stage3'])
            s3_input_dim = dim_state2 + self.stage_dims['stage3']
            self.glu4 = GatedFusion(hist_dim=dim_state3 + self.hd['s1'][-1], curr_dim=self.stage_dims['stage4'])
            s4_input_dim = dim_state3 + self.hd['s1'][-1] + self.stage_dims['stage4']

        elif fusion_type == 'attention':
            d_k = 32
            self.attn2 = AttentionFusion(hist_dims=[dim_state1], curr_dim=self.stage_dims['stage2'], d_k=d_k)
            s2_input_dim = d_k + self.stage_dims['stage2']
            self.attn3 = AttentionFusion(hist_dims=[dim_state1, dim_state2], curr_dim=self.stage_dims['stage3'], d_k=d_k)
            s3_input_dim = d_k + self.stage_dims['stage3']
            self.attn4 = AttentionFusion(hist_dims=[dim_state1, dim_state2, dim_state3], curr_dim=self.stage_dims['stage4'], d_k=d_k)
            s4_input_dim = d_k + self.stage_dims['stage4']

        elif fusion_type == 'cross_attention':
            d_model = 16
            n_heads = 4
            self.attn2 = CrossAttentionFusion(hist_dims=[dim_state1], curr_dim=self.stage_dims['stage2'], d_model=d_model, n_heads=n_heads)
            s2_input_dim = d_model + self.stage_dims['stage2']
            self.attn3 = CrossAttentionFusion(hist_dims=[dim_state1, dim_state2], curr_dim=self.stage_dims['stage3'], d_model=d_model, n_heads=n_heads)
            s3_input_dim = d_model + self.stage_dims['stage3']
            self.attn4 = CrossAttentionFusion(hist_dims=[dim_state1, dim_state2, dim_state3], curr_dim=self.stage_dims['stage4'], d_model=d_model, n_heads=n_heads)
            s4_input_dim = d_model + self.stage_dims['stage4']

        else:
            raise ValueError("fusion_type must be 'concat', 'glu', 'attention', or 'cross_attention'")

        self.s2_layers = build_mlp(s2_input_dim, self.hd['s2'], self.dropout_rate)
        self.out_24_25 = nn.Linear(self.hd['s2'][-1], 2)

        self.s3_layers = build_mlp(s3_input_dim, self.hd['s3'], self.dropout_rate)
        self.out_33 = nn.Linear(self.hd['s3'][-1], 1)

        self.s4_layers = build_mlp(s4_input_dim, self.hd['s4'], self.dropout_rate)
        self.out_35 = nn.Linear(self.hd['s4'][-1], 1)

    def _get_stage_feat(self, stage_name, x_cat, x_num):
        embedded = []
        for i, emb in enumerate(self.embeddings[stage_name]):
            embedded.append(emb(x_cat[:, i]))
        if embedded:
            return torch.cat([x_num] + embedded, dim=1)
        return x_num

    def forward(self, x_dict):
        feat_s1 = self._get_stage_feat("stage1", x_dict["stage1"]["cat"], x_dict["stage1"]["num"])
        feat_s2 = self._get_stage_feat("stage2", x_dict["stage2"]["cat"], x_dict["stage2"]["num"])
        feat_s3 = self._get_stage_feat("stage3", x_dict["stage3"]["cat"], x_dict["stage3"]["num"])
        feat_s4 = self._get_stage_feat("stage4", x_dict["stage4"]["cat"], x_dict["stage4"]["num"])

        h1 = self.s1_layers(feat_s1)
        pred_19 = self.out_19(h1)
        state1 = torch.cat([h1, pred_19], dim=1)

        if self.fusion_type == 'concat': h2_in = torch.cat([state1, feat_s2], dim=1)
        elif self.fusion_type == 'glu': h2_in = self.glu2(state1, feat_s2)
        elif self.fusion_type in ['attention', 'cross_attention']: h2_in = self.attn2([state1], feat_s2)

        h2 = self.s2_layers(h2_in)
        pred_24_25 = self.out_24_25(h2)
        state2 = torch.cat([h2, pred_24_25], dim=1)

        if self.fusion_type == 'concat': h3_in = torch.cat([state2, feat_s3], dim=1)
        elif self.fusion_type == 'glu': h3_in = self.glu3(state2, feat_s3)
        elif self.fusion_type in ['attention', 'cross_attention']: h3_in = self.attn3([state1, state2], feat_s3)

        h3 = self.s3_layers(h3_in)
        pred_33 = self.out_33(h3)
        state3 = torch.cat([h3, pred_33], dim=1)

        if self.fusion_type == 'concat': h4_in = torch.cat([state3, feat_s4, h1], dim=1)
        elif self.fusion_type == 'glu': h4_in = self.glu4(torch.cat([state3, h1], dim=1), feat_s4)
        elif self.fusion_type in ['attention', 'cross_attention']: h4_in = self.attn4([state1, state2, state3], feat_s4)

        h4 = self.s4_layers(h4_in)
        pred_35 = self.out_35(h4)

        return {'stage1': pred_19, 'stage2': pred_24_25, 'stage3': pred_33, 'stage4': pred_35}


class PureMLPModel(nn.Module):
    def __init__(self, stage_configs, hidden_dims=None, dropout_rate=0.3):
        super(PureMLPModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.embeddings = nn.ModuleDict()
        total_input_dim = 0

        for stage, cfg in stage_configs.items():
            emb_list = nn.ModuleList([nn.Embedding(num_embeddings=dim, embedding_dim=2) for dim in cfg["cat_dims"]])
            self.embeddings[stage] = emb_list
            total_input_dim += cfg["num_dim"] + len(cfg["cat_dims"]) * 2

        layers = []
        curr_in = total_input_dim
        for out_features in hidden_dims:
            layers.append(nn.Linear(curr_in, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_in = out_features

        layers.append(nn.Linear(curr_in, 5))
        self.mlp = nn.Sequential(*layers)

    def _get_stage_feat(self, stage_name, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings[stage_name])]
        return torch.cat([x_num] + embedded, dim=1) if embedded else x_num

    def forward(self, x_dict):
        feats = [self._get_stage_feat(s, x_dict[s]["cat"], x_dict[s]["num"]) for s in ["stage1", "stage2", "stage3", "stage4"]]
        combined_input = torch.cat(feats, dim=1)
        return self.mlp(combined_input)


class OriginalCascadedModel(nn.Module):
    def __init__(self, stage_configs, hidden_dims=None, dropout_rate=0.0):
        super(OriginalCascadedModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = {'s1': [64, 32], 's2': [32], 's3': [32], 's4': [16]}
        self.hd = hidden_dims
        self.dropout_rate = dropout_rate

        self.embeddings = nn.ModuleDict()
        self.stage_dims = {}
        for stage, cfg in stage_configs.items():
            emb_list = nn.ModuleList([nn.Embedding(num_embeddings=dim, embedding_dim=2) for dim in cfg["cat_dims"]])
            self.embeddings[stage] = emb_list
            self.stage_dims[stage] = cfg["num_dim"] + len(cfg["cat_dims"]) * 2

        def build_mlp(in_features, hidden_dims_list, drop):
            layers = []
            curr_in = in_features
            for out_features in hidden_dims_list:
                layers.append(nn.Linear(curr_in, out_features))
                layers.append(nn.ReLU())
                if drop > 0:
                    layers.append(nn.Dropout(drop))
                curr_in = out_features
            return nn.Sequential(*layers)

        self.s1_layers = build_mlp(self.stage_dims['stage1'], self.hd['s1'], self.dropout_rate)
        self.out_19 = nn.Linear(self.hd['s1'][-1], 1)

        s2_input_dim = self.hd['s1'][-1] + 1 + self.stage_dims['stage2']
        self.s2_layers = build_mlp(s2_input_dim, self.hd['s2'], self.dropout_rate)
        self.out_24_25 = nn.Linear(self.hd['s2'][-1], 2)

        s3_input_dim = self.hd['s2'][-1] + 2 + self.stage_dims['stage3']
        self.s3_layers = build_mlp(s3_input_dim, self.hd['s3'], self.dropout_rate)
        self.out_33 = nn.Linear(self.hd['s3'][-1], 1)

        s4_input_dim = self.hd['s3'][-1] + 1 + self.stage_dims['stage4']
        self.s4_layers = build_mlp(s4_input_dim, self.hd['s4'], self.dropout_rate)
        self.out_35 = nn.Linear(self.hd['s4'][-1], 1)

    def _get_stage_feat(self, stage_name, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings[stage_name])]
        return torch.cat([x_num] + embedded, dim=1) if embedded else x_num

    def forward(self, x_dict):
        feat_s1 = self._get_stage_feat("stage1", x_dict["stage1"]["cat"], x_dict["stage1"]["num"])
        h1 = self.s1_layers(feat_s1)
        pred_19 = self.out_19(h1)

        feat_s2 = self._get_stage_feat("stage2", x_dict["stage2"]["cat"], x_dict["stage2"]["num"])
        h2_in = torch.cat([h1, pred_19, feat_s2], dim=1)
        h2 = self.s2_layers(h2_in)
        pred_24_25 = self.out_24_25(h2)

        feat_s3 = self._get_stage_feat("stage3", x_dict["stage3"]["cat"], x_dict["stage3"]["num"])
        h3_in = torch.cat([h2, pred_24_25, feat_s3], dim=1)
        h3 = self.s3_layers(h3_in)
        pred_33 = self.out_33(h3)

        feat_s4 = self._get_stage_feat("stage4", x_dict["stage4"]["cat"], x_dict["stage4"]["num"])
        h4_in = torch.cat([h3, pred_33, feat_s4], dim=1)
        h4 = self.s4_layers(h4_in)
        pred_35 = self.out_35(h4)

        return {'stage1': pred_19, 'stage2': pred_24_25, 'stage3': pred_33, 'stage4': pred_35}
