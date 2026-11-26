"""版本化策略预设。前瞻账本、今日变化和解释卡都引用这里的身份证。

预设生效后不修改。规则要变时新建版本，记录生效日期；旧账仍按旧版本解释。
侧栏探索参数不在这里，也不写公共账本。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams

EFFECTIVE = "2026-11-26"


@dataclass(frozen=True)
class V14Preset:
    preset_id: str
    version: int
    effective: str
    entry_z: float
    exit_z: float = 0.5
    stop_z: float = 4.0
    hedge: str = "rolling"
    hedge_window: int = 120
    z_window: int = 60
    kalman_delta_auto: bool = True
    fill: str = "next_close"
    beta_band: float = 0.10
    allow_short_a: bool = False
    engine: str = "v14"

    def strategy_params(self) -> StrategyParams:
        return StrategyParams(
            hedge=self.hedge,
            hedge_window=self.hedge_window,
            z_window=self.z_window,
            entry_z=self.entry_z,
            exit_z=self.exit_z,
            stop_z=self.stop_z,
            kalman_delta_auto=self.kalman_delta_auto,
            allow_short_a=self.allow_short_a,
            fill=self.fill,
            beta_band=self.beta_band,
            costs=CostParams(),
        )

    def spec(self) -> dict:
        return {
            "engine": self.engine,
            "preset_id": self.preset_id,
            "version": self.version,
            "effective": self.effective,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "stop_z": self.stop_z,
            "hedge": self.hedge,
            "hedge_window": self.hedge_window,
            "z_window": self.z_window,
            "kalman_delta_auto": self.kalman_delta_auto,
            "fill": self.fill,
            "beta_band": self.beta_band,
            "allow_short_a": self.allow_short_a,
        }


@dataclass(frozen=True)
class V15Preset:
    preset_id: str
    version: int
    effective: str
    z_in_grid: tuple = (1.5, 2.0, 2.5)
    z_out_grid: tuple = (0.0, 0.5, 1.0)
    w_est_grid: tuple = (120, 250)
    train: int = 756
    embargo: int = 121
    lock_len: int = 126
    z_window: int = 40
    max_hold: int = 60
    stop_z: float = 3.5
    inner_train: int = 500
    inner_valid: int = 256
    min_trades: int = 6
    allow_short_a: bool = False
    engine: str = "v15"

    def spec(self) -> dict:
        return {
            "engine": self.engine,
            "preset_id": self.preset_id,
            "version": self.version,
            "effective": self.effective,
            "z_in_grid": list(self.z_in_grid),
            "z_out_grid": list(self.z_out_grid),
            "w_est_grid": list(self.w_est_grid),
            "train": self.train,
            "embargo": self.embargo,
            "lock_len": self.lock_len,
            "z_window": self.z_window,
            "max_hold": self.max_hold,
            "stop_z": self.stop_z,
            "inner_train": self.inner_train,
            "inner_valid": self.inner_valid,
            "min_trades": self.min_trades,
            "allow_short_a": self.allow_short_a,
        }


PRESETS: dict[str, V14Preset | V15Preset] = {
    "v14-default": V14Preset("v14-default", 1, EFFECTIVE, entry_z=2.0),
    "v14-strict": V14Preset("v14-strict", 1, EFFECTIVE, entry_z=2.5),
    "v15-default": V15Preset("v15-default", 1, EFFECTIVE),
    "v15-strict": V15Preset("v15-strict", 1, EFFECTIVE, z_in_grid=(2.5,)),
}


def get_preset(preset_id: str) -> V14Preset | V15Preset:
    if preset_id not in PRESETS:
        raise KeyError(f"未知预设 {preset_id}")
    return PRESETS[preset_id]


def presets_for(engine: str) -> list:
    return [preset for preset in PRESETS.values() if preset.engine == engine]


def spec_fingerprint(preset) -> str:
    import hashlib
    import json

    payload = json.dumps(preset.spec(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
