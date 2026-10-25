"""Prompt changes within one continuous shot keep the self-attention state."""

def preserve_smooth_context(kv_caches, crossattn_caches):
    if len(kv_caches) != len(crossattn_caches):
        raise ValueError("Self- and cross-attention layer counts must match")
    for cache, text_cache in zip(kv_caches, crossattn_caches):
        # Refresh text conditioning only. Retain PC history, RoPE origin, and
        # local/global pointers, so the next block continues the same shot.
        text_cache["is_init"] = False
        cache["scene_cut"] = False
        cache["active_transition_mode"] = "smooth"
