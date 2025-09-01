from dataclasses import dataclass

@dataclass
class WorkerSpec:
    url: str
    free_vram_mb: int
    start: int = 0
    end: int = 0
