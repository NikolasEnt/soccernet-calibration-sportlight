import json
from typing import Any, Dict, List, Tuple


def read_json(annot_path: str) -> Dict[str, Any]:
    data = {}
    with open(annot_path, 'r') as f:
        data = json.load(f)
    return data


def decode_annot(annot) -> Dict[str, List[Tuple[float, float]]]:
    res: Dict[str, List[Tuple[float, float]]] = {}
    for cls_id in annot:
        res[cls_id] = []
        for point in annot[cls_id]:
            res[cls_id].append((point['x'], point['y']))
    return res


def read_annot(annot_path: str) -> Dict[str, List[Tuple[float, float]]]:
    annot = read_json(annot_path)
    res = decode_annot(annot)
    return res
