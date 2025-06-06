from typing import List, Dict, Tuple
import numpy as np

class FigureMapper:
    def __init__(self):
        """Figure 매퍼 초기화"""
        pass
    
    def map_references_to_figures(self, 
                                 figure_references: List[Dict], 
                                 figure_layouts: List[Dict],
                                 page_index: int) -> List[Dict]:
        """Figure 참조를 실제 Figure 레이아웃에 맵핑"""
        mapped_references = []
        
        for ref in figure_references:
            figure_num = ref['figure_number']
            
            # 동일 페이지에서 Figure 찾기
            matched_figure = self._find_matching_figure(
                figure_num, 
                figure_layouts, 
                page_index
            )
            
            if matched_figure:
                ref['mapped_figure_id'] = matched_figure['figure_id']
                ref['mapped_figure_bbox'] = matched_figure['bbox']
            else:
                ref['mapped_figure_id'] = None
                ref['mapped_figure_bbox'] = None
            
            mapped_references.append(ref)
        
        return mapped_references
    
    def _find_matching_figure(self, 
                            figure_num: int, 
                            figure_layouts: List[Dict],
                            page_index: int) -> Dict:
        """Figure 번호에 해당하는 Figure 레이아웃 찾기"""
        # 같은 페이지의 Figure들 필터링
        page_figures = [
            fig for fig in figure_layouts 
            if fig['page_index'] == page_index
        ]
        
        # Figure 번호 기반 매칭 (간단한 휴리스틱)
        # 실제로는 더 복잡한 매칭 알고리즘 필요
        if 0 < figure_num <= len(page_figures):
            return page_figures[figure_num - 1]
        
        # 위치 기반 매칭 시도
        # TODO: 더 정교한 매칭 알고리즘 구현
        
        return None
    
    def calculate_spatial_relationship(self, 
                                     text_bbox: List[float], 
                                     figure_bbox: List[float]) -> Dict:
        """텍스트와 Figure 간의 공간적 관계 계산"""
        # 중심점 계산
        text_center = self._get_bbox_center(text_bbox)
        figure_center = self._get_bbox_center(figure_bbox)
        
        # 거리 계산
        distance = np.sqrt(
            (text_center[0] - figure_center[0])**2 + 
            (text_center[1] - figure_center[1])**2
        )
        
        # 상대 위치
        position = {
            'above': text_center[1] < figure_center[1],
            'below': text_center[1] > figure_center[1],
            'left': text_center[0] < figure_center[0],
            'right': text_center[0] > figure_center[0]
        }
        
        return {
            'distance': float(distance),
            'position': position,
            'text_center': text_center,
            'figure_center': figure_center
        }
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """바운딩 박스의 중심점 계산"""
        if len(bbox) == 4:  # [x1, y1, x2, y2]
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        elif len(bbox) == 8:  # [x1, y1, x2, y2, x3, y3, x4, y4]
            x_coords = [bbox[i] for i in range(0, 8, 2)]
            y_coords = [bbox[i] for i in range(1, 8, 2)]
            return (sum(x_coords) / 4, sum(y_coords) / 4)
        else:
            return (0, 0)