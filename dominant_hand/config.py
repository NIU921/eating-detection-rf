from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    fs: int = 25
    window_s: int = 10
    stride_s: int = 5
    boundary_s: int = 60
    positive_overlap: float = 0.8
    min_observed: float = 0.95
    max_interpolation_ms: int = 200
    lowpass_hz: float = 8.0
    gravity_hz: float = 0.3
    trim_s: int = 3
    anchor: str = "start"
    seed: int = 20261021

    def __post_init__(self):
        for name in ("fs","window_s","stride_s","boundary_s","max_interpolation_ms","trim_s","seed"):
            if type(getattr(self,name)) is not int:
                raise ValueError(f"{name}必须为整数")
        if self.fs not in (25, 50) or self.anchor not in ("start", "end"):
            raise ValueError("fs只支持25/50Hz，anchor只支持start/end")
        if self.stride_s <= 0 or self.window_s < self.stride_s:
            raise ValueError("要求0 < stride_s <= window_s")
        if self.boundary_s < 0 or not 0 < self.positive_overlap <= 1:
            raise ValueError("边界缓冲或正例覆盖参数无效")
        if not 0 < self.min_observed <= 1 or self.trim_s < 0:
            raise ValueError("质量参数无效")
        if self.max_interpolation_ms < 0 or not 0 < self.gravity_hz < self.lowpass_hz < self.fs/2:
            raise ValueError("滤波及插值参数无效")

    def to_dict(self):
        return asdict(self)
