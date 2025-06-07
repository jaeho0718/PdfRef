import numpy as np
from typing import List, Union

def ensure_bbox_format(bbox: Union[List, np.ndarray]) -> List[float]:
    """bbox를 [xmin, ymin, xmax, ymax] 형식으로 보장"""
    if not isinstance(bbox, (list, np.ndarray)):
        return [0.0, 0.0, 0.0, 0.0]
    
    bbox = np.array(bbox)
    
    # 다각형 형태인 경우 [[x,y], [x,y], ...]
    if bbox.ndim == 2:
        if bbox.shape[0] >= 4 and bbox.shape[1] == 2:
            xmin = float(bbox[:, 0].min())
            ymin = float(bbox[:, 1].min())
            xmax = float(bbox[:, 0].max())
            ymax = float(bbox[:, 1].max())
            return [xmin, ymin, xmax, ymax]
    
    # 이미 bbox 형태인 경우 [xmin, ymin, xmax, ymax]
    elif bbox.ndim == 1 and len(bbox) >= 4:
        return [float(x) for x in bbox[:4]]
    
    return [0.0, 0.0, 0.0, 0.0]

def poly_to_bbox(poly: Union[List, np.ndarray]) -> List[float]:
    """다각형 좌표를 바운딩 박스로 변환"""
    return ensure_bbox_format(poly)