class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, d_k: int = 128,
                 block_size: int = 512, pos_mode: str = "onehot",
                 pos_P: int = 64, num_labels: int | None = None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.use_abs = (pos_mode == "abs")
        self.pos_mode = pos_mode
        self.pos_P = pos_P

        if self.use_abs:
            self.pos_emb = nn.Embedding(block_size, d_model)

        # NEW: paper-style one-hot positions via concat+linear
        if self.pos_mode == "onehot":
            assert self.pos_P is not None, "pos_P (P) required for onehot positions"
            self.in_proj = nn.Linear(d_model + self.pos_P, d_model)

        self.attn1 = SingleHeadCausalAttention(d_model, d_k, pos_mode=("nope" if pos_mode in ["onehot","abs"] else pos_mode))
        self.attn2 = SingleHeadCausalAttention(d_model, d_k, pos_mode=("nope" if pos_mode in ["onehot","abs"] else pos_mode))
        self.ln_f = nn.LayerNorm(d_model)

        # LM head (kept for LM mode)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

        # NEW: optional 3-layer classifier head (paper-style)
        self.classifier = None
        if num_labels is not None:
            h = max(128, d_model)  # small but effective
            self.classifier = nn.Sequential(
                nn.Linear(d_model, h), nn.ReLU(inplace=True),
                nn.Linear(h, h),       nn.ReLU(inplace=True),
                nn.Linear(h, num_labels)
            )

    def forward(self, idx, pos_ids: torch.LongTensor | None = None,
                collect_attn: bool = False, cls: bool = False):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)

        if self.use_abs:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]

        # NEW: one-hot absolute positions (concat then project)
        if self.pos_mode == "onehot":
            assert pos_ids is not None, "forward needs pos_ids when pos_mode='onehot'"
            pos_oh = F.one_hot(pos_ids, num_classes=self.pos_P).float()  # (B,T,P)
            x = torch.cat([x, pos_oh], dim=-1)
            x = self.in_proj(x)

        x = self.attn1(x, collect_attn=collect_attn)
        x = self.attn2(x, collect_attn=collect_attn)
        x = self.ln_f(x)

        if cls and (self.classifier is not None):
            # paper: classify using final position’s representation
            return self.classifier(x[:, -1, :])

        # default LM path
        return self.lm_head(x)
