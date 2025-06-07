from typing import List, Dict, Tuple, Optional
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

class FigureMapper:
    def __init__(self):
        """Figure 매퍼 초기화"""
        self.chapter_patterns = [
            r'Chapter\s+(\d+)',
            r'CHAPTER\s+(\d+)',
            r'제\s*(\d+)\s*장',
            r'Section\s+(\d+(?:\.\d+)*)',
            r'(\d+(?:\.\d+)*)\s+\w+',  # 1.2 Introduction 형식
        ]
        
        self.figure_registry = {}  # 전체 문서의 Figure 레지스트리
        self.chapter_info = {}  # 페이지별 챕터 정보
    
    def build_document_structure(self, all_pages_results: List[Dict]):
        """전체 문서 구조 구축
        
        Args:
            all_pages_results: 모든 페이지의 OCR 결과
        """
        current_chapter = None
        current_section = None
        
        for page_result in all_pages_results:
            page_idx = page_result.get('page_index', 0)
            
            # 1. 페이지의 챕터/섹션 정보 사용 또는 추출
            if 'chapter_info' in page_result:
                chapter_section = page_result['chapter_info']
                if chapter_section.get('chapter') is not None:
                    current_chapter = chapter_section.get('chapter')
                    current_section = chapter_section.get('section')
            else:
                # 챕터/섹션 정보 추출 (fallback)
                chapter_section = self._extract_chapter_section(page_result)
                if chapter_section:
                    current_chapter = chapter_section.get('chapter')
                    current_section = chapter_section.get('section')
            
            # 페이지의 챕터 정보 저장
            self.chapter_info[page_idx] = {
                'chapter': current_chapter,
                'section': current_section,
                'page_index': page_idx
            }
            
            # 2. Figure 정보 수집 및 챕터 정보 추가
            for figure in page_result.get('figure_layouts', []):
                figure_id = figure.get('figure_id')
                figure_number = figure.get('figure_number')
                
                if not figure_number:
                    # Figure 번호가 없으면 캡션에서 추출 시도
                    figure_number = self._extract_figure_number_from_layout(
                        figure, page_result.get('recognized_texts', [])
                    )
                
                if figure_number:
                    # Figure 레지스트리에 등록
                    registry_key = f"fig_{figure_number}"
                    if current_chapter:
                        registry_key = f"ch{current_chapter}_fig{figure_number}"
                    
                    self.figure_registry[registry_key] = {
                        'figure_id': figure_id,
                        'figure_number': figure_number,
                        'chapter': current_chapter,
                        'section': current_section,
                        'page_index': page_idx,
                        'bbox': figure.get('bbox'),
                        'caption': figure.get('caption')
                    }
    
    def _extract_chapter_section(self, page_result: Dict) -> Optional[Dict]:
        """페이지에서 챕터/섹션 정보 추출"""
        texts = page_result.get('recognized_texts', [])
        
        for text_info in texts:
            text = text_info.get('text', '')
            
            # 챕터 패턴 매칭
            for pattern in self.chapter_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # 제목 레이아웃인지 확인 (크기, 위치 등)
                    if self._is_likely_heading(text_info, page_result):
                        chapter_num = match.group(1)
                        
                        # 섹션 번호 분리
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
    
    def _is_likely_heading(self, text_info: Dict, page_result: Dict) -> bool:
        """텍스트가 제목일 가능성 판단"""
        # 레이아웃 정보 활용
        layouts = page_result.get('layouts', [])
        text_bbox = text_info.get('bbox', [])
        
        for layout in layouts:
            if layout.get('label') in ['title', 'paragraph_title', 'header']:
                # 텍스트가 제목 레이아웃 내에 있는지 확인
                if self._is_bbox_inside(text_bbox, layout.get('bbox', [])):
                    return True
        
        return False
    
    def _extract_figure_number_from_layout(self, 
                                         figure: Dict, 
                                         texts: List[Dict]) -> Optional[int]:
        """Figure 레이아웃에서 번호 추출"""
        # Figure 캡션 찾기
        caption = self._find_figure_caption(figure, texts)
        
        if caption:
            # 캡션에서 Figure 번호 추출
            patterns = [
                r'Fig(?:ure)?\.?\s*(\d+)',
                r'그림\s*(\d+)',
                r'Figure\s*(\d+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, caption, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return None
    
    def _find_figure_caption(self, figure: Dict, texts: List[Dict]) -> Optional[str]:
        """Figure의 캡션 텍스트 찾기"""
        figure_bbox = figure.get('bbox', [])
        
        # Figure 아래쪽 텍스트 찾기 (캡션은 주로 아래에 위치)
        caption_candidates = []
        
        for text in texts:
            text_bbox = text.get('bbox', [])
            if not text_bbox:
                continue
            
            # Figure 아래에 있고, 수평적으로 정렬된 텍스트
            if (text_bbox[1] > figure_bbox[3] and  # 아래에 위치
                text_bbox[1] - figure_bbox[3] < 50):  # 너무 멀지 않음
                
                # 수평 정렬 확인
                overlap = self._calculate_horizontal_overlap(figure_bbox, text_bbox)
                if overlap > 0.5:  # 50% 이상 겹침
                    caption_candidates.append({
                        'text': text.get('text', ''),
                        'distance': text_bbox[1] - figure_bbox[3],
                        'overlap': overlap
                    })
        
        # 가장 가까운 캡션 선택
        if caption_candidates:
            caption_candidates.sort(key=lambda x: x['distance'])
            return caption_candidates[0]['text']
        
        return None
    
    def map_references_to_figures(self, 
                                 figure_references: List[Dict], 
                                 current_page: int) -> List[Dict]:
        """개선된 Figure 참조 매핑
        
        Args:
            figure_references: Figure 참조 리스트
            current_page: 현재 페이지 번호
        """
        mapped_references = []
        current_chapter_info = self.chapter_info.get(current_page, {})
        
        for ref in figure_references:
            figure_num = ref.get('figure_number')
            if not figure_num:
                ref['mapped_figure_id'] = None
                mapped_references.append(ref)
                continue
            
            # 1. 먼저 같은 챕터 내에서 찾기
            matched_figure = self._find_figure_in_chapter(
                figure_num, 
                current_chapter_info.get('chapter')
            )
            
            # 2. 못 찾으면 전체 문서에서 찾기
            if not matched_figure:
                matched_figure = self._find_figure_globally(figure_num)
            
            # 3. 그래도 못 찾으면 휴리스틱 사용
            if not matched_figure:
                matched_figure = self._find_figure_heuristic(
                    figure_num, current_page
                )
            
            if matched_figure:
                ref['mapped_figure_id'] = matched_figure.get('figure_id')
                ref['mapped_figure_bbox'] = matched_figure.get('bbox')
                ref['mapped_figure_page'] = matched_figure.get('page_index')
                ref['mapped_chapter'] = matched_figure.get('chapter')
                ref['mapping_confidence'] = self._calculate_mapping_confidence(
                    ref, matched_figure, current_chapter_info
                )
            else:
                ref['mapped_figure_id'] = None
                ref['mapping_confidence'] = 0.0
            
            mapped_references.append(ref)
        
        return mapped_references
    
    def _find_figure_in_chapter(self, 
                               figure_num: int, 
                               chapter: Optional[int]) -> Optional[Dict]:
        """같은 챕터 내에서 Figure 찾기"""
        if chapter is None:
            return None
        
        # 챕터별 Figure 키
        chapter_key = f"ch{chapter}_fig{figure_num}"
        if chapter_key in self.figure_registry:
            return self.figure_registry[chapter_key]
        
        # 챕터 정보 없이 저장된 경우도 확인
        for key, figure in self.figure_registry.items():
            if (figure.get('figure_number') == figure_num and 
                figure.get('chapter') == chapter):
                return figure
        
        return None
    
    def _find_figure_globally(self, figure_num: int) -> Optional[Dict]:
        """전체 문서에서 Figure 찾기"""
        # 정확한 번호 매칭
        simple_key = f"fig_{figure_num}"
        if simple_key in self.figure_registry:
            return self.figure_registry[simple_key]
        
        # 모든 Figure 중에서 번호로 찾기
        for key, figure in self.figure_registry.items():
            if figure.get('figure_number') == figure_num:
                return figure
        
        return None
    
    def _find_figure_heuristic(self, 
                              figure_num: int, 
                              current_page: int) -> Optional[Dict]:
        """휴리스틱을 사용한 Figure 찾기"""
        # 모든 Figure를 페이지 순서로 정렬
        all_figures = sorted(
            self.figure_registry.values(),
            key=lambda f: (f.get('chapter', 0), f.get('page_index', 0))
        )
        
        # Figure 번호가 순차적이라고 가정
        if 0 < figure_num <= len(all_figures):
            return all_figures[figure_num - 1]
        
        return None
    
    def _calculate_mapping_confidence(self, 
                                    reference: Dict, 
                                    figure: Dict,
                                    current_chapter_info: Dict) -> float:
        """매핑 신뢰도 계산"""
        confidence = 0.5
        
        # 같은 챕터면 신뢰도 증가
        if (current_chapter_info.get('chapter') == figure.get('chapter') and
            current_chapter_info.get('chapter') is not None):
            confidence += 0.3
        
        # Figure 번호가 캡션에서 명확히 추출된 경우
        if figure.get('caption'):
            confidence += 0.2
        
        # 페이지 거리 고려
        page_distance = abs(
            reference.get('page_index', 0) - 
            figure.get('page_index', 0)
        )
        
        # 같은 챕터 내에서는 페이지 거리가 덜 중요
        if current_chapter_info.get('chapter') == figure.get('chapter'):
            confidence -= page_distance * 0.02
        else:
            confidence -= page_distance * 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def _is_bbox_inside(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """inner_bbox가 outer_bbox 내부에 있는지 확인"""
        if not isinstance(inner_bbox, (list, np.ndarray)) or len(inner_bbox) < 4:
            return False
        if not isinstance(outer_bbox, (list, np.ndarray)) or len(outer_bbox) < 4:
            return False

        try:
            inner = [float(x) for x in inner_bbox[:4]]
            outer = [float(x) for x in outer_bbox[:4]]
        except (ValueError, TypeError):
            return False

        return (inner[0] >= outer[0] and 
                inner[1] >= outer[1] and 
                inner[2] <= outer[2] and 
                inner[3] <= outer[3])

    def _calculate_horizontal_overlap(self, 
                                    bbox1: List[float], 
                                    bbox2: List[float]) -> float:
        """두 바운딩 박스의 수평 겹침 비율 계산"""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_min, x1_max = bbox1[0], bbox1[2]
        x2_min, x2_max = bbox2[0], bbox2[2]
        
        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)
        
        if overlap_start < overlap_end:
            overlap_width = overlap_end - overlap_start
            bbox1_width = x1_max - x1_min
            bbox2_width = x2_max - x2_min
            min_width = min(bbox1_width, bbox2_width)
            
            return overlap_width / min_width if min_width > 0 else 0
        
        return 0.0