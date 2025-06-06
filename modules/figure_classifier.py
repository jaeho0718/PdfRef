import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Tuple
from config import config

class FigureClassifier:
    def __init__(self, model_name: str = None):
        """Figure 참조 분류기 초기화"""
        self.model_name = model_name or config.BERT_MODEL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 간단한 규칙 기반 분류기 사용 (실제로는 fine-tuned BERT 모델 사용 권장)
        self.figure_patterns = config.FIGURE_PATTERNS
        
        # BERT 모델 로드 (옵션)
        # self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        # self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        # self.model.to(self.device)
        # self.model.eval()
    
    def extract_figure_references(self, texts: List[Dict]) -> List[Dict]:
        """텍스트에서 Figure 참조 추출"""
        figure_references = []
        
        for text_info in texts:
            text = text_info['text']
            
            # 규칙 기반으로 Figure 참조 찾기
            for pattern in self.figure_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Figure 번호 추출
                    figure_num = self._extract_figure_number(match.group())
                    
                    if figure_num is not None:
                        ref_info = {
                            'text': match.group(),
                            'figure_number': figure_num,
                            'text_id': text_info['text_id'],
                            'bbox': text_info['bbox'],
                            'start_pos': match.start(),
                            'end_pos': match.end(),
                            'full_text': text,
                            'confidence': 0.95  # 규칙 기반이므로 높은 신뢰도
                        }
                        figure_references.append(ref_info)
        
        return figure_references
    
    def _extract_figure_number(self, text: str) -> int:
        """텍스트에서 Figure 번호 추출"""
        try:
            # 숫자 추출
            numbers = re.findall(r'\d+', text)
            if numbers:
                return int(numbers[0])
        except:
            pass
        return None
    
    def classify_with_bert(self, texts: List[str]) -> List[Dict]:
        """BERT를 사용한 Figure 참조 분류 (구현 예시)"""
        # 실제 구현시 fine-tuned 모델 사용
        results = []
        
        # for text in texts:
        #     inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #     
        #     with torch.no_grad():
        #         outputs = self.model(**inputs)
        #         probs = torch.softmax(outputs.logits, dim=-1)
        #         
        #     is_figure_ref = probs[0][1] > 0.5  # 임계값
        #     
        #     results.append({
        #         'text': text,
        #         'is_figure_reference': is_figure_ref,
        #         'confidence': float(probs[0][1])
        #     })
        
        return results