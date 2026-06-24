import re
import random
import numpy as np
import torch


def NAME_TO_WIDTH(name: str) -> float:
    """根据模型名称返回对应的 width_mult。
    
    支持格式：mn10_as, dymn20_as, dymn20_as(1) 等。
    解析失败时打印警告并返回 1.0。
    """
    mn_map = {
        'mn01': 0.1, 'mn02': 0.2, 'mn04': 0.4, 'mn05': 0.5,
        'mn06': 0.6, 'mn08': 0.8, 'mn10': 1.0, 'mn12': 1.2,
        'mn14': 1.4, 'mn16': 1.6, 'mn20': 2.0, 'mn30': 3.0, 'mn40': 4.0,
    }
    dymn_map = {'dymn04': 0.4, 'dymn10': 1.0, 'dymn20': 2.0}

    try:
        if name.startswith('dymn'):
            # 匹配 dymn + 数字，如 dymn4、dymn10、dymn20
            m = re.match(r'^(dymn\d+)', name)
            if m:
                key = m.group(1)
                # 规范化：dymn4 → dymn04
                num = re.search(r'\d+', key).group()
                key_norm = 'dymn' + num.zfill(2)
                if key_norm in dymn_map:
                    return dymn_map[key_norm]
        else:
            # 匹配 mn + 数字，如 mn4、mn10、mn40
            m = re.match(r'^(mn\d+)', name)
            if m:
                key = m.group(1)
                num = re.search(r'\d+', key).group()
                key_norm = 'mn' + num.zfill(2)
                if key_norm in mn_map:
                    return mn_map[key_norm]
    except Exception:
        pass

    print(f"警告: NAME_TO_WIDTH 无法识别模型名称 '{name}'，返回默认值 1.0")
    return 1.0

def worker_init_fn(worker_id: int):
    """为每个 DataLoader worker 设置独立的随机种子。"""
    seed_seq = np.random.SeedSequence([torch.initial_seed(), worker_id])
    torch.random.manual_seed(_spawn_int(seed_seq))
    np.random.seed(_spawn_array(seed_seq))
    random.seed(_spawn_int(seed_seq))


def _spawn_int(seed_seq: np.random.SeedSequence) -> int:
    """从 SeedSequence 派生一个整数种子。"""
    a, b = seed_seq.spawn(1)[0].generate_state(2, dtype=np.uint32)
    return int(a) | (int(b) << 32)


def _spawn_array(seed_seq: np.random.SeedSequence) -> np.ndarray:
    """从 SeedSequence 派生一个 uint32 数组种子。"""
    return seed_seq.spawn(1)[0].generate_state(2, dtype=np.uint32)
