import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

from config import config

logger = logging.getLogger(__name__)

class FigureReferenceBERT(nn.Module):
    """Figure Reference 분류를 위한 커스텀 BERT 모델"""
    def __init__(self, bert_model_name: str, num_labels: int = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Figure 번호 추출을 위한 추가 레이어
        self.figure_num_extractor = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 분류 결과
        logits = self.classifier(pooled_output)
        
        # Figure 번호 예측 (회귀)
        figure_num_logits = self.figure_num_extractor(pooled_output)
        
        return {
            'logits': logits,
            'figure_num_logits': figure_num_logits,
            'hidden_states': outputs.last_hidden_state
        }

class FigureClassifier:
    def __init__(self, model_name: str = None, use_bert: bool = True):
        """Figure 참조 분류기 초기화
        
        Args:
            model_name: BERT 모델 이름
            use_bert: BERT 사용 여부 (False면 규칙 기반만 사용)
        """
        self.model_name = model_name or config.BERT_MODEL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_bert = use_bert and torch.cuda.is_available()  # GPU 없으면 규칙 기반만 사용
        
        # 규칙 기반 패턴
        self.figure_patterns = config.FIGURE_PATTERNS
        self.extended_patterns = [
            # 영어 패턴
            r'(?:as\s+shown\s+in\s+)?Fig(?:ure)?\.?\s*(\d+(?:\.\d+)?)',
            r'(?:see\s+)?Figure\s*(\d+(?:\.\d+)?)',
            r'(?:in\s+)?fig(?:ure)?\.?\s*(\d+(?:\.\d+)?)',
            r'\(Fig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\)',
            r'Figures?\s*(\d+(?:\.\d+)?)\s*(?:and|,|through|to|-)\s*(\d+(?:\.\d+)?)',
            # 한국어 패턴
            r'그림\s*(\d+(?:\.\d+)?)',
            r'도표\s*(\d+(?:\.\d+)?)',
            r'\[그림\s*(\d+(?:\.\d+)?)\]',
        ]
        
        if self.use_bert:
            try:
                # BERT 모델 및 토크나이저 로드
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                
                # 사전 훈련된 Figure Reference 모델이 있는지 확인
                model_path = f"models/figure_ref_bert_{self.model_name.replace('/', '_')}.pt"
                
                if os.path.exists(model_path):
                    # 사전 훈련된 모델 로드
                    logger.info(f"Loading pre-trained Figure Reference BERT from {model_path}")
                    self.model = FigureReferenceBERT(self.model_name)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                else:
                    # 기본 BERT 모델로 초기화
                    logger.warning("No pre-trained Figure Reference model found. Using base BERT model.")
                    logger.warning("For better performance, please fine-tune the model on figure reference data.")
                    self.model = BertForSequenceClassification.from_pretrained(
                        self.model_name,
                        num_labels=2  # 0: Not Figure Reference, 1: Figure Reference
                    )
                
                self.model.to(self.device)
                self.model.eval()
                
                # 컨텍스트 윈도우 크기
                self.context_window = 50  # 참조 전후 50자
                
                logger.info(f"BERT model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load BERT model: {e}")
                logger.warning("Falling back to rule-based approach")
                self.use_bert = False
    
    def extract_figure_references(self, texts: List[Dict]) -> List[Dict]:
        """텍스트에서 Figure 참조 추출 (하이브리드 접근법)"""
        figure_references = []
        
        # 1. 규칙 기반 추출
        rule_based_refs = self._extract_with_rules(texts)
        
        # 2. BERT 기반 추출 (사용 가능한 경우)
        if self.use_bert:
            bert_based_refs = self._extract_with_bert(texts)
            
            # 3. 결과 병합 및 중복 제거
            figure_references = self._merge_results(rule_based_refs, bert_based_refs)
        else:
            figure_references = rule_based_refs
        
        return figure_references
    
    def _extract_with_rules(self, texts: List[Dict]) -> List[Dict]:
        """규칙 기반 Figure 참조 추출"""
        figure_references = []
        
        for text_info in texts:
            text = text_info['text']
            
            # 모든 패턴으로 매칭
            all_patterns = self.figure_patterns + self.extended_patterns
            
            for pattern in all_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    # Figure 번호 추출
                    figure_nums = self._extract_all_figure_numbers(match)
                    
                    for figure_num in figure_nums:
                        if figure_num is not None:
                            ref_info = {
                                'text': match.group(),
                                'figure_number': figure_num,
                                'text_id': text_info['text_id'],
                                'bbox': text_info['bbox'],
                                'start_pos': match.start(),
                                'end_pos': match.end(),
                                'full_text': text,
                                'confidence': 0.95,  # 규칙 기반 높은 신뢰도
                                'method': 'rule_based',
                                'context': self._get_context(text, match.start(), match.end())
                            }
                            figure_references.append(ref_info)
        
        return figure_references
    
    def _extract_with_bert(self, texts: List[Dict]) -> List[Dict]:
        """BERT 기반 Figure 참조 추출"""
        figure_references = []
        
        # 배치 처리를 위한 준비
        batch_texts = []
        batch_info = []
        
        for text_info in texts:
            text = text_info['text']
            
            # 문장 단위로 분할 (또는 슬라이딩 윈도우)
            sentences = self._split_into_segments(text)
            
            for sent_start, sent_end, sentence in sentences:
                batch_texts.append(sentence)
                batch_info.append({
                    'text_info': text_info,
                    'start_pos': sent_start,
                    'end_pos': sent_end,
                    'sentence': sentence
                })
        
        # 배치 단위로 처리
        batch_size = 32
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i:i + batch_size]
            batch_meta = batch_info[i:i + batch_size]
            
            # BERT 추론
            predictions = self._bert_inference(batch)
            
            # 결과 처리
            for j, (pred, meta) in enumerate(zip(predictions, batch_meta)):
                if pred['is_figure_ref']:
                    # Figure 번호 추출
                    figure_num = self._extract_figure_number_bert(
                        meta['sentence'], 
                        pred.get('figure_num_pred')
                    )
                    
                    if figure_num is not None:
                        ref_info = {
                            'text': self._extract_reference_text(meta['sentence']),
                            'figure_number': figure_num,
                            'text_id': meta['text_info']['text_id'],
                            'bbox': meta['text_info']['bbox'],
                            'start_pos': meta['start_pos'],
                            'end_pos': meta['end_pos'],
                            'full_text': meta['text_info']['text'],
                            'confidence': float(pred['confidence']),
                            'method': 'bert',
                            'context': meta['sentence']
                        }
                        figure_references.append(ref_info)
        
        return figure_references
    
    def _bert_inference(self, texts: List[str]) -> List[Dict]:
        """BERT 모델 추론"""
        # 토큰화
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # GPU로 이동
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 추론
        with torch.no_grad():
            if isinstance(self.model, FigureReferenceBERT):
                outputs = self.model(**encoded)
                logits = outputs['logits']
                figure_num_logits = outputs.get('figure_num_logits')
            else:
                outputs = self.model(**encoded)
                logits = outputs.logits
                figure_num_logits = None
            
            # 확률 계산
            probs = torch.softmax(logits, dim=-1)
            
            # 결과 처리
            predictions = []
            for i in range(len(texts)):
                is_figure_ref = probs[i][1] > 0.5  # 클래스 1이 Figure Reference
                confidence = float(probs[i][1] if is_figure_ref else probs[i][0])
                
                pred = {
                    'is_figure_ref': bool(is_figure_ref),
                    'confidence': confidence
                }
                
                # Figure 번호 예측값 추가
                if figure_num_logits is not None and is_figure_ref:
                    pred['figure_num_pred'] = float(figure_num_logits[i])
                
                predictions.append(pred)
        
        return predictions
    
    def _split_into_segments(self, text: str, max_length: int = 100) -> List[Tuple[int, int, str]]:
        """텍스트를 세그먼트로 분할"""
        segments = []
        
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]\s+', text)
        
        current_pos = 0
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 원본 텍스트에서 문장 위치 찾기
            start_pos = text.find(sentence, current_pos)
            if start_pos == -1:
                continue
            
            end_pos = start_pos + len(sentence)
            
            # 문장이 너무 길면 추가 분할
            if len(sentence) > max_length:
                # 슬라이딩 윈도우 방식
                for i in range(0, len(sentence), max_length // 2):
                    sub_start = i
                    sub_end = min(i + max_length, len(sentence))
                    sub_text = sentence[sub_start:sub_end]
                    
                    segments.append((
                        start_pos + sub_start,
                        start_pos + sub_end,
                        sub_text
                    ))
            else:
                segments.append((start_pos, end_pos, sentence))
            
            current_pos = end_pos
        
        return segments
    
    def _extract_all_figure_numbers(self, match) -> List[int]:
        """매치에서 모든 Figure 번호 추출"""
        numbers = []
        
        # 모든 그룹에서 숫자 추출
        for i in range(1, len(match.groups()) + 1):
            try:
                group = match.group(i)
                if group:
                    # 소수점 처리
                    if '.' in group:
                        num = float(group)
                        numbers.append(int(num))  # 정수 부분만
                    else:
                        numbers.append(int(group))
            except:
                continue
        
        return numbers if numbers else [None]
    
    def _extract_figure_number_bert(self, text: str, pred_num: Optional[float] = None) -> Optional[int]:
        """BERT 예측 결과와 텍스트에서 Figure 번호 추출"""
        # 먼저 규칙 기반으로 시도
        for pattern in self.extended_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).split('.')[0])
                except:
                    continue
        
        # BERT 예측값 사용 (있는 경우)
        if pred_num is not None and pred_num > 0:
            return int(round(pred_num))
        
        return None
    
    def _extract_reference_text(self, sentence: str) -> str:
        """문장에서 Figure 참조 텍스트 추출"""
        # Figure 참조 패턴 찾기
        for pattern in self.extended_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                # 매치 주변 컨텍스트 포함
                start = max(0, match.start() - 10)
                end = min(len(sentence), match.end() + 10)
                return sentence[start:end].strip()
        
        # 못 찾으면 전체 문장 반환
        return sentence.strip()
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """참조 주변 컨텍스트 추출"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _merge_results(self, rule_based: List[Dict], bert_based: List[Dict]) -> List[Dict]:
        """규칙 기반과 BERT 기반 결과 병합"""
        merged = []
        
        # 규칙 기반 결과를 먼저 추가
        for ref in rule_based:
            ref['final_confidence'] = ref['confidence']
            merged.append(ref)
        
        # BERT 기반 결과 추가 (중복 확인)
        for bert_ref in bert_based:
            is_duplicate = False
            
            for rule_ref in merged:
                # 같은 텍스트 영역에서 같은 Figure 번호를 참조하는지 확인
                if (rule_ref['text_id'] == bert_ref['text_id'] and 
                    rule_ref['figure_number'] == bert_ref['figure_number'] and
                    abs(rule_ref['start_pos'] - bert_ref['start_pos']) < 20):
                    
                    # 신뢰도가 더 높은 것으로 업데이트
                    if bert_ref['confidence'] > rule_ref.get('final_confidence', 0):
                        rule_ref['final_confidence'] = bert_ref['confidence']
                        rule_ref['method'] = 'hybrid'
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                bert_ref['final_confidence'] = bert_ref['confidence'] * 0.9  # BERT만 검출한 경우 약간 낮은 신뢰도
                merged.append(bert_ref)
        
        # 신뢰도 순으로 정렬
        merged.sort(key=lambda x: x.get('final_confidence', 0), reverse=True)
        
        return merged
    
    def fine_tune_model(self, training_data: List[Dict], epochs: int = 3):
        """Figure Reference 분류 모델 fine-tuning (별도 구현 필요)"""
        # 이 메서드는 학습 데이터가 있을 때 모델을 fine-tuning하는 용도
        # 실제 구현은 별도의 학습 스크립트로 분리하는 것이 좋음
        logger.warning("Fine-tuning requires labeled training data and is not implemented in this version.")
        logger.warning("Please prepare training data in format: {'text': str, 'is_figure_ref': bool, 'figure_number': int}")
        pass