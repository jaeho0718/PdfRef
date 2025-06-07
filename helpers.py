import os
from typing import List, Dict, Any
import json
import cv2
import numpy as np

def save_results_to_json(results: Dict[str, Any], output_path: str):
    """결과를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def visualize_results(image_path: str, results: Dict[str, Any], output_path: str):
    """결과 시각화"""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    # Figure 레이아웃 그리기 (파란색)
    for figure in results.get('figure_layouts', []):
        bbox = figure['bbox']
        cv2.rectangle(image, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (255, 0, 0), 2)
        cv2.putText(image, figure['figure_id'], 
                   (int(bbox[0]), int(bbox[1])-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Figure 참조 텍스트 그리기 (빨간색)
    for ref in results.get('figure_references', []):
        if ref['mapped_figure_id']:
            bbox = ref['bbox']
            if len(bbox) >= 4:
                cv2.rectangle(image,
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             (0, 0, 255), 2)
                cv2.putText(image, ref['text'],
                           (int(bbox[0]), int(bbox[1])-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, image)

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """파일 확장자 검증"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions