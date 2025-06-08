from typing import List, Dict, Any, Tuple, Callable, Optional
import os
import tempfile
import shutil
import logging
from datetime import datetime
import re
import numpy as np
import concurrent.futures

from .layout_detector import LayoutDetector
from .text_detector import TextDetector
from .figure_classifier import FigureClassifier
from .figure_mapper import FigureMapper
from .pdf_processor import PDFProcessor
from .parallel_processor import ParallelProcessor
from .response_models import transform_analysis_result

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self):
        """문서 분석기 초기화"""
        self.layout_detector = LayoutDetector()
        self.text_detector = TextDetector()
        self.figure_classifier = FigureClassifier()
        self.figure_mapper = FigureMapper()
        self.pdf_processor = PDFProcessor()
        self.parallel_processor = ParallelProcessor()
        
    def analyze_pdf(self, pdf_path: str, 
                   chunk_size: int = 10,
                   progress_callback: Callable = None,
                   frontend_format: bool = False) -> Dict[str, Any]:
        """PDF 문서 전체 분석
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 처리 청크 크기
            progress_callback: 진행 상황 콜백
            frontend_format: True이면 프론트엔드 친화적인 형식으로 반환
        """
        start_time = datetime.now()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # PDF 정보 추출
            pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
            total_pages = pdf_info['total_pages']
            
            logger.info(f"PDF 분석 시작: {total_pages} 페이지")
            
            # PDF를 이미지로 변환
            pages = list(self.pdf_processor.convert_pdf_to_images(pdf_path))
            
            # 병렬 처리로 페이지별 분석
            results = self.parallel_processor.process_pages_parallel(
                pages,
                self._analyze_single_page_with_size,
                batch_size=1
            )
            
            # 문서 구조 구축 (챕터/섹션 정보 포함)
            self.figure_mapper.build_document_structure(results)
            
            # 각 페이지의 Figure 참조 재매핑
            for result in results:
                if 'error' not in result and 'figure_references' in result:
                    page_idx = result['page_index']
                    # 문서 구조를 활용한 매핑
                    result['figure_references'] = self.figure_mapper.map_references_to_figures(
                        result['figure_references'],
                        page_idx
                    )
            
            # 최종 결과 구성
            analysis_result = {
                'status': 'success',
                'pdf_info': pdf_info,
                'total_pages': total_pages,
                'pages': results,
                'document_structure': {
                    'chapters': self._extract_chapter_summary(),
                    'figure_registry': self.figure_mapper.figure_registry
                },
                'summary': self._generate_summary(results),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            # 프론트엔드 형식으로 변환
            if frontend_format:
                return transform_analysis_result(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PDF 분석 실패: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def analyze_pdf_with_callbacks(self, 
                                  pdf_path: str,
                                  chunk_size: int = 10,
                                  progress_callback: Callable = None,
                                  page_callback: Callable = None,
                                  frontend_format: bool = False) -> Dict[str, Any]:
        """콜백과 함께 PDF 분석 (CLI용)
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 처리 청크 크기
            progress_callback: 진행 상황 콜백
            page_callback: 페이지 완료 콜백
            frontend_format: True이면 프론트엔드 친화적인 형식으로 반환
        """
        start_time = datetime.now()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # PDF 정보 추출
            pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
            total_pages = pdf_info['total_pages']
            
            # PDF를 이미지로 변환
            pages = list(self.pdf_processor.convert_pdf_to_images(pdf_path))
            
            # 병렬 처리 with 콜백
            results = self.parallel_processor.process_pages_with_callbacks(
                pages,
                self._analyze_single_page_with_size,
                progress_callback=progress_callback,
                page_callback=page_callback
            )
            
            # 문서 구조 구축
            self.figure_mapper.build_document_structure(results)
            
            # Figure 참조 재매핑
            for result in results:
                if 'error' not in result and 'figure_references' in result:
                    page_idx = result['page_index']
                    result['figure_references'] = self.figure_mapper.map_references_to_figures(
                        result['figure_references'],
                        page_idx
                    )
            
            # 최종 결과 구성
            analysis_result = {
                'status': 'success',
                'pdf_info': pdf_info,
                'total_pages': total_pages,
                'pages': results,
                'document_structure': {
                    'chapters': self._extract_chapter_summary(),
                    'figure_registry': self.figure_mapper.figure_registry
                },
                'summary': self._generate_summary(results),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            # 프론트엔드 형식으로 변환
            if frontend_format:
                return transform_analysis_result(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PDF 분석 실패: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _analyze_single_page_with_size(self, page_data: Tuple, *args) -> Dict[str, Any]:
        """단일 페이지 분석 - 크기 정보 포함"""
        if len(page_data) == 4:
            page_index, image_path, width, height = page_data
            size = {"width": width, "height": height}
        else:
            # 기존 형식과의 호환성
            page_index, image_path = page_data[:2]
            size = None
        
        result = self._analyze_single_page(image_path, page_index)
        if result.get('status') == 'success' and size:
            result['size'] = size
        
        return result
    
    def _analyze_single_page(self, image_path: str, page_index: int) -> Dict[str, Any]:
        """단일 페이지 분석 - 챕터/섹션 정보 포함"""
        try:
            # 1. 레이아웃 감지
            layout_result = self.layout_detector.detect_layout(image_path, page_index)
            
            # 2. 텍스트 영역 필터링
            text_regions = self.layout_detector.filter_text_regions(
                layout_result['layouts']
            )
            
            # 3. 텍스트 감지 및 인식
            recognized_texts = self.text_detector.detect_and_recognize(
                image_path, 
                text_regions
            )
            
            # 4. 챕터/섹션 정보 추출
            chapter_info = self._extract_page_structure_info(
                recognized_texts, 
                layout_result['layouts']
            )
            
            # 5. Figure 레이아웃에 추가 정보 보강
            enhanced_figure_layouts = self._enhance_figure_layouts(
                layout_result['figure_layouts'],
                recognized_texts,
                chapter_info
            )
            
            # 6. Figure 참조 추출
            figure_references = self.figure_classifier.extract_figure_references(
                recognized_texts
            )
            
            # 7. 페이지 컨텍스트 정보 추가
            for ref in figure_references:
                ref['page_index'] = page_index
                ref['chapter'] = chapter_info.get('chapter')
                ref['section'] = chapter_info.get('section')
            
            return {
                'page_index': page_index,
                'layouts': layout_result['layouts'],
                'figure_layouts': enhanced_figure_layouts,
                'recognized_texts': recognized_texts,
                'figure_references': figure_references,
                'chapter_info': chapter_info,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"페이지 {page_index} 분석 실패: {str(e)}")
            return {
                'page_index': page_index,
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_page_structure_info(self, 
                                texts: List[Dict], 
                                layouts: List[Dict]) -> Dict[str, Any]:
        """페이지에서 구조 정보 (챕터, 섹션 등) 추출"""
        chapter_info = {
            'chapter': None,
            'section': None,
            'subsection': None,
            'title': None
        }

        # 제목 레이아웃 찾기
        title_layouts = [
            layout for layout in layouts 
            if layout.get('label', '').lower() in ['title', 'paragraph_title', 'header']
        ]

        # 각 텍스트 검사
        for text_info in texts:
            text = text_info.get('text', '').strip()
            bbox = self._normalize_bbox(text_info.get('bbox', []))

            # 텍스트가 제목 레이아웃 내에 있는지 확인
            is_title = False
            for title_layout in title_layouts:
                if self._is_bbox_inside(bbox, title_layout.get('bbox', [])):
                    is_title = True
                    break
                
            # 챕터 패턴 매칭
            chapter_match = self._match_chapter_pattern(text)
            if chapter_match and (is_title or self._is_likely_chapter_heading(text_info, texts)):
                chapter_info.update(chapter_match)
                continue
            
            # 섹션 패턴 매칭
            section_match = self._match_section_pattern(text)
            if section_match and is_title:
                chapter_info['section'] = section_match.get('section')
                chapter_info['subsection'] = section_match.get('subsection')

        return chapter_info
    
    def _match_chapter_pattern(self, text: str) -> Optional[Dict]:
        """챕터 패턴 매칭"""
        patterns = [
            (r'Chapter\s+(\d+)(?:\s*[:\.]?\s*(.*))?', 'en'),
            (r'CHAPTER\s+(\d+)(?:\s*[:\.]?\s*(.*))?', 'en'),
            (r'제\s*(\d+)\s*장(?:\s*[:\.]?\s*(.*))?', 'ko'),
            (r'第\s*(\d+)\s*章(?:\s*[:\.]?\s*(.*))?', 'zh'),
        ]
        
        for pattern, lang in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'chapter': int(match.group(1)),
                    'title': match.group(2).strip() if match.group(2) else None,
                    'language': lang
                }
        
        return None
    
    def _match_section_pattern(self, text: str) -> Optional[Dict]:
        """섹션 패턴 매칭"""
        patterns = [
            r'(\d+)\.(\d+)(?:\.(\d+))?\s*(.*)',  # 1.2.3 형식
            r'Section\s+(\d+)(?:\.(\d+))?\s*(.*)',
            r'§\s*(\d+)(?:\.(\d+))?\s*(.*)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    'section': f"{groups[0]}.{groups[1]}" if groups[1] else groups[0],
                    'subsection': groups[2] if len(groups) > 2 and groups[2] else None,
                    'title': groups[-1].strip() if groups[-1] else None
                }
        
        return None
    
    def _normalize_bbox(self, bbox):
        """bbox를 [x_min, y_min, x_max, y_max] 형식으로 정규화"""
        if not bbox or len(bbox) == 0:
            return []

        # 이미 [x_min, y_min, x_max, y_max] 형식인 경우
        if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            return bbox

        # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형식인 경우
        if len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        # 그 외의 경우
        return []

    def _is_likely_chapter_heading(self, text_info: Dict, all_texts: List[Dict]) -> bool:
        """텍스트가 챕터 제목일 가능성 판단"""
        text = text_info.get('text', '')
        bbox = text_info.get('bbox', [])
        
        # bbox가 리스트인지 확인하고 적절한 길이인지 체크
        if not isinstance(bbox, (list, np.ndarray)) or len(bbox) < 4:
            return False
        
        # 각 요소가 숫자인지 확인
        try:
            bbox_values = [float(x) for x in bbox]
        except (ValueError, TypeError):
            return False
        
        # 휴리스틱 1: 페이지 상단에 위치
        if bbox_values[1] < 200:  # Y 좌표가 200 픽셀 이내
            return True
        
        # 휴리스틱 2: 다른 텍스트보다 큰 폰트 (bbox 높이로 추정)
        text_height = bbox_values[3] - bbox_values[1]
        
        # 평균 높이 계산
        heights = []
        for t in all_texts:
            t_bbox = t.get('bbox', [])
            if isinstance(t_bbox, (list, np.ndarray)) and len(t_bbox) >= 4:
                try:
                    t_bbox_values = [float(x) for x in t_bbox]
                    heights.append(t_bbox_values[3] - t_bbox_values[1])
                except:
                    continue
                
        if heights:
            avg_height = np.mean(heights)
            if text_height > avg_height * 1.5:
                return True
        
        # 휴리스틱 3: 독립된 줄 (주변에 다른 텍스트 없음)
        if self._is_isolated_text(text_info, all_texts):
            return True
        
        return False

    def _is_isolated_text(self, text_info: Dict, all_texts: List[Dict]) -> bool:
        """텍스트가 독립된 줄인지 확인"""
        bbox = self._normalize_bbox(text_info.get('bbox', []))
        if len(bbox) < 4:
            return False

        # 같은 Y 범위에 있는 다른 텍스트 찾기
        y_center = (bbox[1] + bbox[3]) / 2
        y_tolerance = (bbox[3] - bbox[1]) / 2

        nearby_texts = 0
        for other in all_texts:
            if other == text_info:
                continue
            
            other_bbox = self._normalize_bbox(other.get('bbox', []))
            if len(other_bbox) >= 4:
                other_y_center = (other_bbox[1] + other_bbox[3]) / 2
                if abs(other_y_center - y_center) < y_tolerance:
                    nearby_texts += 1

        return nearby_texts == 0

    def _is_bbox_inside(self, inner_bbox, outer_bbox) -> bool:
        """inner_bbox가 outer_bbox 내부에 있는지 확인"""
        inner = self._normalize_bbox(inner_bbox)
        outer = self._normalize_bbox(outer_bbox)

        if len(inner) < 4 or len(outer) < 4:
            return False

        # 약간의 여유를 두고 확인 (OCR 오차 고려)
        margin = 5
        return (inner[0] >= outer[0] - margin and 
                inner[1] >= outer[1] - margin and 
                inner[2] <= outer[2] + margin and 
                inner[3] <= outer[3] + margin)

    def _calculate_caption_position_score(self, figure_bbox, text_bbox) -> float:
        """캡션 위치 점수 계산"""
        # bbox 정규화
        figure_bbox = self._normalize_bbox(figure_bbox)
        text_bbox = self._normalize_bbox(text_bbox)

        if len(figure_bbox) < 4 or len(text_bbox) < 4:
            return 0.0

        # Figure 중심과 텍스트 중심
        fig_center_x = (figure_bbox[0] + figure_bbox[2]) / 2
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2

        # 수평 정렬 점수
        horizontal_distance = abs(fig_center_x - text_center_x)
        fig_width = figure_bbox[2] - figure_bbox[0]
        horizontal_score = max(0, 1 - horizontal_distance / fig_width) if fig_width > 0 else 0

        # 수직 거리 점수
        vertical_distance = 0
        if text_bbox[1] > figure_bbox[3]:  # 텍스트가 아래에
            vertical_distance = text_bbox[1] - figure_bbox[3]
        elif text_bbox[3] < figure_bbox[1]:  # 텍스트가 위에
            vertical_distance = figure_bbox[1] - text_bbox[3]
        else:
            return 0  # 겹치는 경우

        # 너무 멀면 점수 감소
        vertical_score = max(0, 1 - vertical_distance / 100)

        # 최종 점수
        return horizontal_score * 0.7 + vertical_score * 0.3

    
    def _enhance_figure_layouts(self, 
                              figure_layouts: List[Dict],
                              texts: List[Dict],
                              chapter_info: Dict) -> List[Dict]:
        """Figure 레이아웃에 추가 정보 보강"""
        enhanced_layouts = []
        
        for idx, figure in enumerate(figure_layouts):
            enhanced_figure = figure.copy()
            
            # 챕터 정보 추가
            enhanced_figure['chapter'] = chapter_info.get('chapter')
            enhanced_figure['section'] = chapter_info.get('section')
            
            # Figure 캡션 찾기 및 번호 추출
            caption_info = self._find_and_parse_figure_caption(figure, texts)
            if caption_info:
                enhanced_figure['caption'] = caption_info.get('text')
                enhanced_figure['figure_number'] = caption_info.get('number')
                enhanced_figure['caption_bbox'] = caption_info.get('bbox')
            
            # Figure ID 개선 (챕터 정보 포함)
            if enhanced_figure.get('figure_number'):
                if chapter_info.get('chapter'):
                    enhanced_figure['figure_id'] = (
                        f"ch{chapter_info['chapter']}_fig{enhanced_figure['figure_number']}"
                    )
                else:
                    enhanced_figure['figure_id'] = f"fig_{enhanced_figure['figure_number']}"
            
            enhanced_layouts.append(enhanced_figure)
        
        return enhanced_layouts
    
    def _find_and_parse_figure_caption(self, 
                                     figure: Dict, 
                                     texts: List[Dict]) -> Optional[Dict]:
        """Figure 캡션 찾기 및 파싱"""
        figure_bbox = figure.get('bbox', [])
        if len(figure_bbox) < 4:
            return None
        
        # Figure 아래/위의 텍스트 찾기
        caption_candidates = []
        
        for text in texts:
            text_content = text.get('text', '').strip()
            text_bbox = text.get('bbox', [])
            
            if not text_content or len(text_bbox) < 4:
                continue
            
            # Figure 캡션 패턴 확인
            caption_patterns = [
                r'Fig(?:ure)?\.?\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
                r'Figure\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
                r'그림\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
                r'图\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
            ]
            
            for pattern in caption_patterns:
                match = re.match(pattern, text_content, re.IGNORECASE)
                if match:
                    # 위치 관계 확인
                    position_score = self._calculate_caption_position_score(
                        figure_bbox, text_bbox
                    )
                    
                    if position_score > 0.5:  # 임계값
                        caption_candidates.append({
                            'text': text_content,
                            'number': int(match.group(1)),
                            'title': match.group(2).strip() if match.group(2) else None,
                            'bbox': text_bbox,
                            'score': position_score
                        })
        
        # 가장 좋은 캡션 선택
        if caption_candidates:
            caption_candidates.sort(key=lambda x: x['score'], reverse=True)
            return caption_candidates[0]
        
        return None
    
    def _extract_chapter_summary(self) -> List[Dict]:
        """챕터 구조 요약"""
        chapters = {}
        
        for page_idx, chapter_info in self.figure_mapper.chapter_info.items():
            chapter = chapter_info.get('chapter')
            if chapter:
                if chapter not in chapters:
                    chapters[chapter] = {
                        'chapter': chapter,
                        'start_page': page_idx,
                        'end_page': page_idx,
                        'sections': set(),
                        'figure_count': 0
                    }
                else:
                    chapters[chapter]['end_page'] = max(
                        chapters[chapter]['end_page'], 
                        page_idx
                    )
                
                section = chapter_info.get('section')
                if section:
                    chapters[chapter]['sections'].add(section)
        
        # Figure 수 계산
        for fig_info in self.figure_mapper.figure_registry.values():
            chapter = fig_info.get('chapter')
            if chapter and chapter in chapters:
                chapters[chapter]['figure_count'] += 1
        
        # 정렬 및 변환
        chapter_list = []
        for chapter in sorted(chapters.keys()):
            chapter_info = chapters[chapter]
            chapter_info['sections'] = sorted(list(chapter_info['sections']))
            chapter_list.append(chapter_info)
        
        return chapter_list
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """분석 결과 요약 생성"""
        total_layouts = 0
        total_figures = 0
        total_texts = 0
        total_references = 0
        error_pages = 0
        
        for result in results:
            if result.get('status') == 'error':
                error_pages += 1
                continue
            
            total_layouts += len(result.get('layouts', []))
            total_figures += len(result.get('figure_layouts', []))
            total_texts += len(result.get('recognized_texts', []))
            total_references += len(result.get('figure_references', []))
        
        return {
            'total_layouts': total_layouts,
            'total_figures': total_figures,
            'total_texts': total_texts,
            'total_figure_references': total_references,
            'error_pages': error_pages,
            'success_rate': (len(results) - error_pages) / len(results) * 100 if results else 0
        }