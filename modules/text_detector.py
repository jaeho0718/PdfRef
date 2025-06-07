from paddleocr import TextDetection, TextRecognition
import numpy as np
from typing import List, Dict, Tuple, Any
import cv2

class TextDetector:
    def __init__(self, det_model: str = None, rec_model: str = None):
        """텍스트 감지 및 인식기 초기화"""
        self.det_model = det_model or "PP-OCRv5_server_det"
        self.rec_model = rec_model or "PP-OCRv5_server_rec"
        
        self.text_detector = TextDetection(
            model_name=self.det_model,
            device='gpu'
        )
        
        self.text_recognizer = TextRecognition(
            model_name=self.rec_model,
            device='gpu'
        )
    def detect_text(self, image_path: str, text_regions: List[Dict] = None) -> List[Dict]:
        """이미지에서 텍스트 감지"""
        try:
            # 전체 이미지에서 텍스트 감지
            det_output = self.text_detector.predict(image_path, batch_size=1)
            
            text_boxes = []
            for res in det_output:
                result = res.json
                dt_polys = result['res']['dt_polys']
                dt_scores = result['res']['dt_scores']
                
                for poly, score in zip(dt_polys, dt_scores):
                    # 다각형 좌표를 바운딩 박스로 변환
                    bbox = self._poly_to_bbox(poly)
                    text_boxes.append({
                        'bbox': bbox,
                        'poly': poly.tolist() if hasattr(poly, 'tolist') else poly,
                        'score': float(score)
                    })
            
            return text_boxes
            
        except Exception as e:
            print(f"텍스트 감지 오류: {str(e)}")
            return []
    
    def _poly_to_bbox(self, poly):
        """다각형 좌표를 바운딩 박스로 변환
        poly: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태
        return: [xmin, ymin, xmax, ymax] 형태
        """
        if isinstance(poly, (list, np.ndarray)):
            poly = np.array(poly)
            if poly.ndim == 2 and poly.shape[0] >= 4:
                xmin = poly[:, 0].min()
                ymin = poly[:, 1].min()
                xmax = poly[:, 0].max()
                ymax = poly[:, 1].max()
                return [float(xmin), float(ymin), float(xmax), float(ymax)]
        
        # 이미 bbox 형태인 경우
        if isinstance(poly, (list, np.ndarray)) and len(poly) == 4:
            return [float(x) for x in poly]
        
        return [0, 0, 0, 0]  # 기본값
    
    def recognize_text(self, image_path: str, text_boxes: List[Dict]) -> List[Dict]:
        """감지된 텍스트 영역에서 텍스트 인식"""
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            recognized_texts = []
            
            for i, box_info in enumerate(text_boxes):
                # bbox 사용 (이미 변환된 형태)
                bbox = box_info['bbox']
                
                # 바운딩 박스 좌표 확인
                x_min = int(max(0, bbox[0]))
                y_min = int(max(0, bbox[1]))
                x_max = int(min(image.shape[1], bbox[2]))
                y_max = int(min(image.shape[0], bbox[3]))
                
                cropped = image[y_min:y_max, x_min:x_max]
                
                if cropped.size == 0:
                    continue
                
                # 텍스트 인식
                rec_output = self.text_recognizer.predict(cropped, batch_size=1)
                
                for res in rec_output:
                    result = res.json
                    text = result['res']['rec_text']
                    conf = result['res']['rec_score']
                    
                    recognized_texts.append({
                        'text': text,
                        'bbox': bbox,
                        'poly': box_info.get('poly', []),
                        'score': conf,
                        'det_score': box_info['score'],
                        'text_id': f"text_{i}"
                    })
            
            return recognized_texts
            
        except Exception as e:
            print(f"텍스트 인식 오류: {str(e)}")
            return []
        
    def detect_and_recognize(self, image_path: str, text_regions: List[Dict] = None) -> List[Dict]:
        """텍스트 감지 및 인식 통합 수행"""
        text_boxes = self.detect_text(image_path, text_regions)
        recognized_texts = self.recognize_text(image_path, text_boxes)
        return recognized_texts