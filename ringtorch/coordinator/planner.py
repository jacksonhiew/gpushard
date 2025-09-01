from typing import List, Dict


def plan_contiguous(workers: List[Dict], n_layers: int) -> List[Dict]:
    total = sum(w["free_vram_mb"] for w in workers)
    assignments = []
    start = 0
    for i, w in enumerate(workers):
        if i == len(workers) - 1:
            end = n_layers
        else:
            share = max(1, round(n_layers * w["free_vram_mb"] / total))
            end = min(n_layers, start + share)
        assignments.append({**w, "start": start, "end": end})
        start = end
    return assignments
