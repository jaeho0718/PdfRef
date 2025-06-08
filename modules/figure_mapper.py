from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import re
import logging
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)

class ImprovedFigureMapper:
    def __init__(self):
        """개선된 Figure 매퍼 초기화"""
        # 다양한 Figure 패턴 지원
        self.figure_patterns = [
            # 영어 패턴
            r'Fig(?:ure)?\.?\s*(\d+(?:\.\d+)?)',
            r'Figure\s+(\d+(?:\.\d+)?)',
            r'FIG\.?\s*(\d+(?:\.\d+)?)',
            # 한국어 패턴
            r'그림\s*(\d+(?:\.\d+)?)',
            r'도\s*(\d+(?:\.\d+)?)',
            # 중국어 패턴
            r'图\s*(\d+(?:\.\d+)?)',
            r'圖\s*(\d+(?:\.\d+)?)',
        ]
        
        self.chapter_patterns = [
            r'Chapter\s+(\d+)',
            r'CHAPTER\s+(\d+)',
            r'제\s*(\d+)\s*장',
            r'Section\s+(\d+(?:\.\d+)*)',
            r'(\d+(?:\.\d+)*)\s+\w+',
        ]
        
        # 데이터 구조
        self.figure_registry = {}  # 모든 Figure 정보
        self.figure_by_number = defaultdict(list)  # 번호별 Figure 리스트
        self.figure_by_page = defaultdict(list)  # 페이지별 Figure 리스트
        self.chapter_info = {}  # 페이지별 챕터 정보
        self.caption_texts = {}  # Figure ID별 캡션 텍스트
        
        # 매핑 통계
        self.mapping_stats = {
            'total_figures': 0,
            'total_references': 0,
            'successful_mappings': 0,
            'failed_mappings': 0,
            'mapping_methods': defaultdict(int)
        }
    
    def build_document_structure(self, all_pages_results: List[Dict]):
        """전체 문서 구조 구축 - 개선된 버전"""
        logger.info("문서 구조 구축 시작")
        
        current_chapter = None
        current_section = None
        
        for page_result in all_pages_results:
            page_idx = page_result.get('page_index', 0)
            
            # 1. 챕터/섹션 정보 업데이트
            chapter_section = self._extract_or_use_chapter_info(
                page_result, current_chapter, current_section
            )
            if chapter_section:
                current_chapter = chapter_section.get('chapter', current_chapter)
                current_section = chapter_section.get('section', current_section)
            
            self.chapter_info[page_idx] = {
                'chapter': current_chapter,
                'section': current_section,
                'page_index': page_idx
            }
            
            # 2. Figure 정보 수집 및 개선
            self._process_page_figures(page_result, current_chapter, current_section)
        
        # 3. Figure 번호 후처리 (누락된 번호 추정)
        self._post_process_figure_numbers()
        
        logger.info(f"문서 구조 구축 완료: {self.mapping_stats['total_figures']} figures found")
        self._print_figure_registry_summary()
    
    def _extract_or_use_chapter_info(self, page_result: Dict, 
                                   current_chapter: Optional[int], 
                                   current_section: Optional[str]) -> Optional[Dict]:
        """챕터/섹션 정보 추출 또는 사용"""
        # 페이지에 이미 챕터 정보가 있으면 사용
        if 'chapter_info' in page_result:
            return page_result['chapter_info']
        
        # 없으면 텍스트에서 추출
        texts = page_result.get('recognized_texts', [])
        for text_info in texts:
            text = text_info.get('text', '')
            
            # 챕터 패턴 매칭
            for pattern in self.chapter_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    chapter_num = match.group(1)
                    if '.' in str(chapter_num):
                        parts = str(chapter_num).split('.')
                        return {
                            'chapter': int(parts[0]),
                            'section': '.'.join(parts[1:]) if len(parts) > 1 else None,
                            'text': text
                        }
                    else:
                        return {
                            'chapter': int(chapter_num),
                            'section': None,
                            'text': text
                        }
        
        return None
    
    def _process_page_figures(self, page_result: Dict, 
                            chapter: Optional[int], 
                            section: Optional[str]):
        """페이지의 Figure 처리 - 개선된 버전"""
        page_idx = page_result.get('page_index', 0)
        
        for figure in page_result.get('figure_layouts', []):
            figure_id = figure.get('figure_id')
            
            # 1. Figure 번호 추출 (여러 방법 시도)
            figure_number = self._extract_figure_number_comprehensive(
                figure, page_result
            )
            
            # 2. Figure 정보 저장
            figure_info = {
                'figure_id': figure_id,
                'figure_number': figure_number,
                'chapter': chapter,
                'section': section,
                'page_index': page_idx,
                'bbox': figure.get('bbox'),
                'caption': figure.get('caption'),
                'caption_bbox': figure.get('caption_bbox'),
                'confidence': figure.get('confidence', 0.5)
            }
            
            # 3. 다양한 인덱스에 저장
            self._register_figure(figure_info)
            self.mapping_stats['total_figures'] += 1
    
    def _extract_figure_number_comprehensive(self, figure: Dict, 
                                           page_result: Dict) -> Optional[int]:
        """Figure 번호 추출 - 여러 방법 시도"""
        # 1. 이미 번호가 있으면 사용
        if figure.get('figure_number'):
            return figure.get('figure_number')
        
        # 2. 캡션에서 추출
        if figure.get('caption'):
            number = self._extract_number_from_text(figure['caption'])
            if number:
                return number
        
        # 3. Figure 근처 텍스트에서 추출
        number = self._find_figure_number_nearby(
            figure, page_result.get('recognized_texts', [])
        )
        if number:
            return number
        
        # 4. Figure ID에서 추출 시도
        if figure.get('figure_id'):
            match = re.search(r'fig_?(\d+)', figure['figure_id'], re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_number_from_text(self, text: str) -> Optional[int]:
        """텍스트에서 Figure 번호 추출"""
        for pattern in self.figure_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # 소수점이 있으면 정수 부분만
                    number_str = match.group(1)
                    if '.' in number_str:
                        return int(float(number_str))
                    return int(number_str)
                except:
                    continue
        return None
    
    def _find_figure_number_nearby(self, figure: Dict, 
                                 texts: List[Dict]) -> Optional[int]:
        """Figure 근처 텍스트에서 번호 찾기"""
        figure_bbox = figure.get('bbox', [])
        if not figure_bbox or len(figure_bbox) < 4:
            return None
        
        # Figure 주변 텍스트 찾기 (위, 아래, 좌, 우)
        nearby_texts = []
        for text in texts:
            text_bbox = text.get('bbox', [])
            if not text_bbox or len(text_bbox) < 4:
                continue
            
            # 거리 계산
            distance = self._calculate_bbox_distance(figure_bbox, text_bbox)
            if distance < 100:  # 100픽셀 이내
                nearby_texts.append({
                    'text': text.get('text', ''),
                    'distance': distance,
                    'bbox': text_bbox
                })
        
        # 가까운 순으로 정렬
        nearby_texts.sort(key=lambda x: x['distance'])
        
        # Figure 패턴 찾기
        for item in nearby_texts[:5]:  # 가장 가까운 5개만
            number = self._extract_number_from_text(item['text'])
            if number:
                return number
        
        return None
    
    def _calculate_bbox_distance(self, bbox1: List[float], 
                               bbox2: List[float]) -> float:
        """두 바운딩 박스 간 최단 거리 계산"""
        # 중심점 간 거리
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def _register_figure(self, figure_info: Dict):
        """Figure를 여러 인덱스에 등록"""
        figure_id = figure_info['figure_id']
        figure_number = figure_info.get('figure_number')
        chapter = figure_info.get('chapter')
        page_idx = figure_info.get('page_index')
        
        # 1. 기본 레지스트리
        self.figure_registry[figure_id] = figure_info
        
        # 2. 번호별 인덱스
        if figure_number:
            self.figure_by_number[figure_number].append(figure_info)
            
            # 캡션 텍스트 저장
            if figure_info.get('caption'):
                self.caption_texts[figure_id] = figure_info['caption']
        
        # 3. 페이지별 인덱스
        self.figure_by_page[page_idx].append(figure_info)
        
        # 4. 챕터별 키도 생성
        if chapter and figure_number:
            chapter_key = f"ch{chapter}_fig{figure_number}"
            self.figure_registry[chapter_key] = figure_info
    
    def _post_process_figure_numbers(self):
        """Figure 번호 후처리 - 누락된 번호 추정"""
        # 페이지 순서대로 Figure 정렬
        all_figures = []
        for page_idx in sorted(self.figure_by_page.keys()):
            all_figures.extend(self.figure_by_page[page_idx])
        
        # 번호가 없는 Figure에 대해 추정
        last_number = 0
        for i, figure in enumerate(all_figures):
            if figure.get('figure_number'):
                last_number = figure['figure_number']
            else:
                # 이전/이후 Figure 번호를 기반으로 추정
                estimated_number = self._estimate_figure_number(
                    i, all_figures, last_number
                )
                if estimated_number:
                    figure['figure_number'] = estimated_number
                    figure['number_estimated'] = True
                    # 인덱스 업데이트
                    self.figure_by_number[estimated_number].append(figure)
    
    def _estimate_figure_number(self, index: int, 
                              all_figures: List[Dict], 
                              last_known: int) -> Optional[int]:
        """Figure 번호 추정"""
        # 간단한 휴리스틱: 순차적 증가 가정
        return last_known + 1 if last_known > 0 else None
    
    def map_references_to_figures(self, 
                                figure_references: List[Dict], 
                                current_page: int) -> List[Dict]:
        """개선된 Figure 참조 매핑"""
        mapped_references = []
        self.mapping_stats['total_references'] += len(figure_references)
        
        for ref in figure_references:
            figure_num = ref.get('figure_number')
            if not figure_num:
                ref['mapped_figure_id'] = None
                ref['mapping_confidence'] = 0.0
                ref['mapping_method'] = 'no_number'
                mapped_references.append(ref)
                self.mapping_stats['failed_mappings'] += 1
                continue
            
            # 다양한 매핑 전략 시도
            mapping_result = self._try_multiple_mapping_strategies(
                ref, figure_num, current_page
            )
            
            if mapping_result['figure']:
                ref['mapped_figure_id'] = mapping_result['figure']['figure_id']
                ref['mapped_figure_bbox'] = mapping_result['figure']['bbox']
                ref['mapped_figure_page'] = mapping_result['figure']['page_index']
                ref['mapping_confidence'] = mapping_result['confidence']
                ref['mapping_method'] = mapping_result['method']
                self.mapping_stats['successful_mappings'] += 1
                self.mapping_stats['mapping_methods'][mapping_result['method']] += 1
            else:
                ref['mapped_figure_id'] = None
                ref['mapping_confidence'] = 0.0
                ref['mapping_method'] = 'failed'
                self.mapping_stats['failed_mappings'] += 1
            
            mapped_references.append(ref)
        
        return mapped_references
    
    def _try_multiple_mapping_strategies(self, 
                                       ref: Dict, 
                                       figure_num: int, 
                                       current_page: int) -> Dict:
        """여러 매핑 전략 시도"""
        current_chapter = self.chapter_info.get(current_page, {}).get('chapter')
        
        strategies = [
            # 1. 같은 챕터 내 정확한 번호 매칭
            ('same_chapter_exact', 
             lambda: self._find_figure_same_chapter(figure_num, current_chapter)),
            
            # 2. 근접 페이지에서 찾기 (±5 페이지)
            ('nearby_pages', 
             lambda: self._find_figure_nearby_pages(figure_num, current_page, 5)),
            
            # 3. 전체 문서에서 정확한 번호 매칭
            ('global_exact', 
             lambda: self._find_figure_by_number(figure_num)),
            
            # 4. 텍스트 유사도 기반 매칭 (캡션 비교)
            ('text_similarity', 
             lambda: self._find_figure_by_text_similarity(ref, figure_num)),
            
            # 5. 순차적 번호 가정 (휴리스틱)
            ('sequential_heuristic', 
             lambda: self._find_figure_sequential(figure_num, current_page))
        ]
        
        for method_name, strategy_func in strategies:
            try:
                result = strategy_func()
                if result:
                    logger.debug(f"Figure {figure_num} mapped using {method_name}")
                    return {
                        'figure': result['figure'],
                        'confidence': result['confidence'],
                        'method': method_name
                    }
            except Exception as e:
                logger.error(f"Error in {method_name}: {str(e)}")
                continue
        
        return {'figure': None, 'confidence': 0.0, 'method': None}
    
    def _find_figure_same_chapter(self, figure_num: int, 
                                chapter: Optional[int]) -> Optional[Dict]:
        """같은 챕터 내에서 Figure 찾기"""
        if not chapter:
            return None
        
        # 챕터별 키로 먼저 시도
        chapter_key = f"ch{chapter}_fig{figure_num}"
        if chapter_key in self.figure_registry:
            return {
                'figure': self.figure_registry[chapter_key],
                'confidence': 0.95
            }
        
        # 번호로 찾은 후 챕터 확인
        candidates = self.figure_by_number.get(figure_num, [])
        for figure in candidates:
            if figure.get('chapter') == chapter:
                return {
                    'figure': figure,
                    'confidence': 0.9
                }
        
        return None
    
    def _find_figure_nearby_pages(self, figure_num: int, 
                                current_page: int, 
                                page_range: int) -> Optional[Dict]:
        """근접 페이지에서 Figure 찾기"""
        candidates = self.figure_by_number.get(figure_num, [])
        
        best_candidate = None
        min_distance = float('inf')
        
        for figure in candidates:
            page_distance = abs(figure['page_index'] - current_page)
            if page_distance <= page_range and page_distance < min_distance:
                min_distance = page_distance
                best_candidate = figure
        
        if best_candidate:
            # 거리에 따른 신뢰도 계산
            confidence = 0.8 - (min_distance / page_range) * 0.3
            return {
                'figure': best_candidate,
                'confidence': confidence
            }
        
        return None
    
    def _find_figure_by_number(self, figure_num: int) -> Optional[Dict]:
        """전체 문서에서 번호로 Figure 찾기"""
        candidates = self.figure_by_number.get(figure_num, [])
        
        if len(candidates) == 1:
            # 유일한 매칭
            return {
                'figure': candidates[0],
                'confidence': 0.7
            }
        elif len(candidates) > 1:
            # 여러 매칭 중 첫 번째 선택 (개선 필요)
            return {
                'figure': candidates[0],
                'confidence': 0.5
            }
        
        return None
    
    def _find_figure_by_text_similarity(self, ref: Dict, 
                                      figure_num: int) -> Optional[Dict]:
        """텍스트 유사도 기반 매칭"""
        ref_text = ref.get('text', '').lower()
        if not ref_text:
            return None
        
        # 참조 텍스트에서 추가 정보 추출 (예: "Figure 3.2 shows...")
        context_words = self._extract_context_words(ref_text)
        if not context_words:
            return None
        
        candidates = self.figure_by_number.get(figure_num, [])
        best_match = None
        best_score = 0.0
        
        for figure in candidates:
            caption = figure.get('caption', '').lower()
            if not caption:
                continue
            
            # 유사도 계산
            score = self._calculate_text_similarity(context_words, caption)
            if score > best_score:
                best_score = score
                best_match = figure
        
        if best_match and best_score > 0.3:
            return {
                'figure': best_match,
                'confidence': min(0.6, best_score)
            }
        
        return None
    
    def _extract_context_words(self, text: str) -> List[str]:
        """참조 텍스트에서 컨텍스트 단어 추출"""
        # Figure 참조 패턴 제거
        for pattern in self.figure_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 불용어 제거 및 중요 단어 추출
        stop_words = {'the', 'a', 'an', 'is', 'are', 'shows', 'displays', 
                     'illustrates', 'depicts', 'in', 'of', 'for'}
        
        words = text.split()
        context_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        
        return context_words[:5]  # 최대 5개 단어
    
    def _calculate_text_similarity(self, words: List[str], 
                                 caption: str) -> float:
        """텍스트 유사도 계산"""
        if not words or not caption:
            return 0.0
        
        # 단어 매칭 수 계산
        matches = sum(1 for word in words if word in caption)
        
        # 유사도 점수
        return matches / len(words)
    
    def _find_figure_sequential(self, figure_num: int, 
                              current_page: int) -> Optional[Dict]:
        """순차적 번호 가정으로 Figure 찾기"""
        # 모든 Figure를 페이지 순으로 정렬
        all_figures = []
        for page_idx in sorted(self.figure_by_page.keys()):
            all_figures.extend(self.figure_by_page[page_idx])
        
        # 번호가 있는 Figure만 필터링
        numbered_figures = [f for f in all_figures if f.get('figure_number')]
        
        if figure_num <= len(numbered_figures):
            # 순차적 번호 가정
            return {
                'figure': numbered_figures[figure_num - 1],
                'confidence': 0.3
            }
        
        return None
    
    def _print_figure_registry_summary(self):
        """Figure 레지스트리 요약 출력 (디버깅용)"""
        logger.info("=== Figure Registry Summary ===")
        logger.info(f"Total figures: {len(self.figure_registry)}")
        logger.info(f"Figures by number: {dict(self.figure_by_number.keys())}")
        
        # 챕터별 Figure 수
        chapter_counts = defaultdict(int)
        for fig in self.figure_registry.values():
            if isinstance(fig, dict) and fig.get('chapter'):
                chapter_counts[fig['chapter']] += 1
        
        logger.info(f"Figures by chapter: {dict(chapter_counts)}")
        
        # 번호가 없는 Figure 수
        no_number_count = sum(
            1 for fig in self.figure_registry.values() 
            if isinstance(fig, dict) and not fig.get('figure_number')
        )
        logger.info(f"Figures without numbers: {no_number_count}")
    
    def get_mapping_statistics(self) -> Dict:
        """매핑 통계 반환"""
        success_rate = (
            self.mapping_stats['successful_mappings'] / 
            self.mapping_stats['total_references'] * 100
            if self.mapping_stats['total_references'] > 0 else 0
        )
        
        return {
            'total_figures': self.mapping_stats['total_figures'],
            'total_references': self.mapping_stats['total_references'],
            'successful_mappings': self.mapping_stats['successful_mappings'],
            'failed_mappings': self.mapping_stats['failed_mappings'],
            'success_rate': success_rate,
            'mapping_methods': dict(self.mapping_stats['mapping_methods'])
        }