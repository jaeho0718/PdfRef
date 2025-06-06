from paddleocr import LayoutDetection
import numpy as np
from typing import List, Dict, Any
import cv2

class LayoutDetector:
    def __init__(self, model_name: str = None):
        """레이아웃 감지기 초기화"""
        self.model_name = model_name or "PP-DocLayout_plus-L"
        self.detector = LayoutDetection(
            model_name=self.model_name,
            device='gpu',
            enable_mkldnn=True
        )
        
    def detect_layout(self, image_path: str, page_index: int = 0) -> Dict[str, Any]:
        """이미지에서 레이아웃 감지"""
        try:
            # 레이아웃 감지 수행
            output = self.detector.predict(image_path, batch_size=1)
            
            layouts = []
            figure_layouts = []
            
            for res in output:
                result = res.json
                boxes = result['res']['boxes']
                
                for box in boxes:
                    layout_info = {
                        'label': box['label'],
                        'bbox': box['coordinate'],
                        'score': box['score'],
                        'page_index': page_index,
                        'cls_id': box['cls_id']
                    }
                    
                    layouts.append(layout_info)
                    
                    # Figure 관련 레이아웃 분류
                    if box['label'].lower() in ['figure', 'image', 'chart', 'diagram']:
                        figure_info = layout_info.copy()
                        figure_info['figure_id'] = f"fig_{page_index}_{len(figure_layouts)}"
                        figure_layouts.append(figure_info)
            
            return {
                'layouts': layouts,
                'figure_layouts': figure_layouts,
                'page_index': page_index
            }
            
        except Exception as e:
            print(f"레이아웃 감지 오류: {str(e)}")
            return {
                'layouts': [],
                'figure_layouts': [],
                'page_index': page_index,
                'error': str(e)
            }
    
    def filter_text_regions(self, layouts: List[Dict]) -> List[Dict]:
        """텍스트 영역만 필터링"""
        text_labels = ['text', 'paragraph_title', 'paragraph', 'title', 'abstract']
        return [
            layout for layout in layouts 
            if layout['label'].lower() in text_labels
        ]