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
from .improved_figure_mapper import ImprovedFigureMapper  # 개선된 버전 사용
from .pdf_processor import PDFProcessor
from .parallel_processor import ParallelProcessor
from .response_models import transform_analysis_result

# 로깅 설정 개선
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, enable_debug_logging: bool = False):
        """문서 분석기 초기화
        
        Args:
            enable_debug_logging: 디버깅 로깅 활성화
        """
        self.layout_detector = LayoutDetector()
        self.text_detector = TextDetector()
        self.figure_classifier = FigureClassifier()
        self.figure_mapper = ImprovedFigureMapper()  # 개선된 매퍼 사용
        self.pdf_processor = PDFProcessor()
        self.parallel_processor = ParallelProcessor()
        
        # 디버깅 모드
        if enable_debug_logging:
            logger.setLevel(logging.DEBUG)
            # Figure mapper 로거도 디버그 모드로
            logging.getLogger('modules.improved_figure_mapper').setLevel(logging.DEBUG)
    
    def analyze_pdf_with_callbacks(self, 
                                  pdf_path: str,
                                  chunk_size: int = 10,
                                  progress_callback: Callable = None,
                                  page_callback: Callable = None,
                                  frontend_format: bool = False) -> Dict[str, Any]:
        """콜백과 함께 PDF 분석 (개선된 Figure 매핑)"""
        start_time = datetime.now()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # PDF 정보 추출
            pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
            total_pages = pdf_info['total_pages']
            
            logger.info(f"PDF 분석 시작: {total_pages} 페이지")
            
            # PDF를 이미지로 변환
            pages = list(self.pdf_processor.convert_pdf_to_images(pdf_path))
            
            # 병렬 처리 with 콜백
            results = self.parallel_processor.process_pages_with_callbacks(
                pages,
                self._analyze_single_page_with_size,
                progress_callback=progress_callback,
                page_callback=page_callback
            )
            
            # 문서 구조 구축 (개선된 버전)
            logger.info("문서 구조 구축 중...")
            self.figure_mapper.build_document_structure(results)
            
            # Figure 참조 재매핑 (개선된 버전)
            logger.info("Figure 참조 매핑 중...")
            total_refs = 0
            mapped_refs = 0
            
            for result in results:
                if 'error' not in result and 'figure_references' in result:
                    page_idx = result['page_index']
                    refs = result['figure_references']
                    total_refs += len(refs)
                    
                    # 개선된 매핑 수행
                    result['figure_references'] = self.figure_mapper.map_references_to_figures(
                        refs, page_idx
                    )
                    
                    # 매핑 성공 수 계산
                    mapped_refs += sum(
                        1 for ref in result['figure_references'] 
                        if ref.get('mapped_figure_id')
                    )
                    
                    # 페이지별 매핑 결과 로깅
                    if refs:
                        page_mapped = sum(
                            1 for ref in result['figure_references'] 
                            if ref.get('mapped_figure_id')
                        )
                        logger.debug(
                            f"Page {page_idx}: {page_mapped}/{len(refs)} references mapped"
                        )
            
            # 매핑 통계 가져오기
            mapping_stats = self.figure_mapper.get_mapping_statistics()
            
            # 매핑 결과 로깅
            logger.info(f"Figure 매핑 완료: {mapped_refs}/{total_refs} 참조 매핑됨")
            logger.info(f"매핑 성공률: {mapping_stats['success_rate']:.1f}%")
            logger.info(f"매핑 방법별 통계: {mapping_stats['mapping_methods']}")
            
            # 최종 결과 구성
            analysis_result = {
                'status': 'success',
                'pdf_info': pdf_info,
                'total_pages': total_pages,
                'pages': results,
                'document_structure': {
                    'chapters': self._extract_chapter_summary(),
                    'figure_registry': dict(self.figure_mapper.figure_registry),
                    'mapping_statistics': mapping_stats  # 매핑 통계 추가
                },
                'summary': self._generate_enhanced_summary(results, mapping_stats),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            # 프론트엔드 형식으로 변환
            if frontend_format:
                return transform_analysis_result(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PDF 분석 실패: {str(e)}", exc_info=True)
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
            page_index, image_path = page_data[:2]
            size = None
        
        result = self._analyze_single_page(image_path, page_index)
        if result.get('status') == 'success' and size:
            result['size'] = size
        
        return result
    
    def _analyze_single_page(self, image_path: str, page_index: int) -> Dict[str, Any]:
        """단일 페이지 분석 - 개선된 Figure 처리"""
        try:
            logger.debug(f"페이지 {page_index} 분석 시작")
            
            # 1. 레이아웃 감지
            layout_result = self.layout_detector.detect_layout(image_path, page_index)
            logger.debug(
                f"페이지 {page_index}: {len(layout_result['layouts'])} 레이아웃, "
                f"{len(layout_result['figure_layouts'])} Figure 감지됨"
            )
            
            # 2. 텍스트 영역 필터링
            text_regions = self.layout_detector.filter_text_regions(
                layout_result['layouts']
            )
            
            # 3. 텍스트 감지 및 인식
            recognized_texts = self.text_detector.detect_and_recognize(
                image_path, 
                text_regions
            )
            logger.debug(f"페이지 {page_index}: {len(recognized_texts)} 텍스트 인식됨")
            
            # 4. 챕터/섹션 정보 추출
            chapter_info = self._extract_page_structure_info(
                recognized_texts, 
                layout_result['layouts']
            )
            if chapter_info.get('chapter'):
                logger.debug(
                    f"페이지 {page_index}: Chapter {chapter_info['chapter']} "
                    f"Section {chapter_info.get('section', 'None')}"
                )
            
            # 5. Figure 레이아웃에 추가 정보 보강
            enhanced_figure_layouts = self._enhance_figure_layouts(
                layout_result['figure_layouts'],
                recognized_texts,
                chapter_info
            )
            
            # Figure 번호 로깅
            for fig in enhanced_figure_layouts:
                if fig.get('figure_number'):
                    logger.debug(
                        f"페이지 {page_index}: Figure {fig['figure_number']} "
                        f"(ID: {fig.get('figure_id')})"
                    )
            
            # 6. Figure 참조 추출
            figure_references = self.figure_classifier.extract_figure_references(
                recognized_texts
            )
            
            # Figure 참조 로깅
            for ref in figure_references:
                logger.debug(
                    f"페이지 {page_index}: Figure 참조 '{ref['text']}' "
                    f"(Figure {ref.get('figure_number')})"
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
            logger.error(f"페이지 {page_index} 분석 실패: {str(e)}", exc_info=True)
            return {
                'page_index': page_index,
                'status': 'error',
                'error': str(e)
            }
    
    def _enhance_figure_layouts(self, 
                              figure_layouts: List[Dict],
                              texts: List[Dict],
                              chapter_info: Dict) -> List[Dict]:
        """Figure 레이아웃에 추가 정보 보강 - 개선된 버전"""
        enhanced_layouts = []
        
        for idx, figure in enumerate(figure_layouts):
            enhanced_figure = figure.copy()
            
            # 챕터 정보 추가
            enhanced_figure['chapter'] = chapter_info.get('chapter')
            enhanced_figure['section'] = chapter_info.get('section')
            
            # Figure 캡션 찾기 및 번호 추출 (개선된 로직)
            caption_info = self._find_and_parse_figure_caption_improved(figure, texts)
            if caption_info:
                enhanced_figure['caption'] = caption_info.get('text')
                enhanced_figure['figure_number'] = caption_info.get('number')
                enhanced_figure['caption_bbox'] = caption_info.get('bbox')
                enhanced_figure['caption_confidence'] = caption_info.get('confidence', 0.5)
                
                logger.debug(
                    f"Figure 캡션 발견: '{caption_info['text']}' "
                    f"(번호: {caption_info.get('number')})"
                )
            
            # Figure ID 개선
            if enhanced_figure.get('figure_number'):
                if chapter_info.get('chapter'):
                    enhanced_figure['figure_id'] = (
                        f"ch{chapter_info['chapter']}_fig{enhanced_figure['figure_number']}"
                    )
                else:
                    enhanced_figure['figure_id'] = f"fig_{enhanced_figure['figure_number']}"
            else:
                # 번호가 없는 경우 페이지와 인덱스 기반 ID
                page_idx = enhanced_figure.get('page_index', 0)
                enhanced_figure['figure_id'] = f"fig_p{page_idx}_{idx}"
            
            enhanced_layouts.append(enhanced_figure)
        
        return enhanced_layouts
    
    def _find_and_parse_figure_caption_improved(self, 
                                              figure: Dict, 
                                              texts: List[Dict]) -> Optional[Dict]:
        """개선된 Figure 캡션 찾기 및 파싱"""
        figure_bbox = figure.get('bbox', [])
        if len(figure_bbox) < 4:
            return None
        
        # Figure 주변의 모든 텍스트 수집 (위, 아래, 좌, 우)
        nearby_texts = []
        
        for text in texts:
            text_content = text.get('text', '').strip()
            text_bbox = text.get('bbox', [])
            
            if not text_content or len(text_bbox) < 4:
                continue
            
            # 거리 계산
            distance = self._calculate_bbox_distance(figure_bbox, text_bbox)
            position = self._get_relative_position(figure_bbox, text_bbox)
            
            nearby_texts.append({
                'text': text_content,
                'bbox': text_bbox,
                'distance': distance,
                'position': position,
                'confidence': text.get('confidence', 0.5)
            })
        
        # 거리순으로 정렬
        nearby_texts.sort(key=lambda x: x['distance'])
        
        # 가장 가까운 텍스트들에서 캡션 패턴 찾기
        caption_patterns = [
            r'Fig(?:ure)?\.?\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
            r'Figure\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
            r'그림\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
            r'图\s*(\d+)(?:\s*[:\.]?\s*(.*))?',
        ]
        
        for candidate in nearby_texts[:10]:  # 가장 가까운 10개만 확인
            text_content = candidate['text']
            
            for pattern in caption_patterns:
                match = re.match(pattern, text_content, re.IGNORECASE)
                if match:
                    # 위치 점수 계산 (캡션은 주로 아래나 위에 위치)
                    position_score = self._calculate_caption_position_score(
                        candidate['position'], candidate['distance']
                    )
                    
                    if position_score > 0.3:  # 임계값
                        return {
                            'text': text_content,
                            'number': int(match.group(1)),
                            'title': match.group(2).strip() if match.group(2) else None,
                            'bbox': candidate['bbox'],
                            'confidence': position_score * candidate['confidence']
                        }
        
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
    
    def _get_relative_position(self, ref_bbox: List[float], 
                             target_bbox: List[float]) -> str:
        """target이 ref에 대해 어느 위치에 있는지 반환"""
        ref_center_y = (ref_bbox[1] + ref_bbox[3]) / 2
        target_center_y = (target_bbox[1] + target_bbox[3]) / 2
        
        if target_center_y < ref_center_y:
            return 'above'
        else:
            return 'below'
    
    def _calculate_caption_position_score(self, position: str, distance: float) -> float:
        """캡션 위치에 대한 점수 계산"""
        # 위치에 따른 기본 점수
        if position in ['below', 'above']:
            base_score = 0.8
        else:
            base_score = 0.3
        
        # 거리에 따른 감쇠
        distance_penalty = min(distance / 200, 1.0)  # 200픽셀을 최대 거리로
        
        return base_score * (1 - distance_penalty * 0.5)
    
    def _extract_chapter_summary(self) -> List[Dict]:
        """챕터 구조 요약 - 개선된 버전"""
        chapters = {}
        
        # 페이지별 챕터 정보 수집
        for page_idx, chapter_info in self.figure_mapper.chapter_info.items():
            chapter = chapter_info.get('chapter')
            if chapter:
                if chapter not in chapters:
                    chapters[chapter] = {
                        'chapter': chapter,
                        'title': None,  # 추후 개선
                        'start_page': page_idx,
                        'end_page': page_idx,
                        'sections': set(),
                        'figure_count': 0,
                        'reference_count': 0
                    }
                else:
                    chapters[chapter]['end_page'] = max(
                        chapters[chapter]['end_page'], 
                        page_idx
                    )
                
                section = chapter_info.get('section')
                if section:
                    chapters[chapter]['sections'].add(section)
        
        # Figure 및 참조 수 계산
        for fig_info in self.figure_mapper.figure_registry.values():
            if isinstance(fig_info, dict):
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
    
    def _generate_enhanced_summary(self, results: List[Dict], 
                                 mapping_stats: Dict) -> Dict[str, Any]:
        """향상된 분석 결과 요약"""
        total_layouts = 0
        total_figures = 0
        total_texts = 0
        total_references = 0
        error_pages = 0
        
        # 기본 통계
        for result in results:
            if result.get('status') == 'error':
                error_pages += 1
                continue
            
            total_layouts += len(result.get('layouts', []))
            total_figures += len(result.get('figure_layouts', []))
            total_texts += len(result.get('recognized_texts', []))
            total_references += len(result.get('figure_references', []))
        
        # Figure 번호 분포
        figure_numbers = []
        for result in results:
            if result.get('status') != 'error':
                for fig in result.get('figure_layouts', []):
                    if fig.get('figure_number'):
                        figure_numbers.append(fig['figure_number'])
        
        return {
            'total_layouts': total_layouts,
            'total_figures': total_figures,
            'total_texts': total_texts,
            'total_figure_references': total_references,
            'mapped_references': mapping_stats['successful_mappings'],
            'unmapped_references': mapping_stats['failed_mappings'],
            'mapping_success_rate': mapping_stats['success_rate'],
            'mapping_methods_used': mapping_stats['mapping_methods'],
            'error_pages': error_pages,
            'success_rate': (len(results) - error_pages) / len(results) * 100 if results else 0,
            'figure_number_range': {
                'min': min(figure_numbers) if figure_numbers else None,
                'max': max(figure_numbers) if figure_numbers else None,
                'unique_count': len(set(figure_numbers))
            }
        }
    
    # 나머지 메서드들은 이전과 동일...
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

        return []

    def _is_likely_chapter_heading(self, text_info: Dict, all_texts: List[Dict]) -> bool:
        """텍스트가 챕터 제목일 가능성 판단"""
        text = text_info.get('text', '')
        bbox = text_info.get('bbox', [])
        
        if not isinstance(bbox, (list, np.ndarray)) or len(bbox) < 4:
            return False
        
        try:
            bbox_values = [float(x) for x in bbox]
        except (ValueError, TypeError):
            return False
        
        # 휴리스틱 1: 페이지 상단에 위치
        if bbox_values[1] < 200:
            return True
        
        # 휴리스틱 2: 다른 텍스트보다 큰 폰트
        text_height = bbox_values[3] - bbox_values[1]
        
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
        
        # 휴리스틱 3: 독립된 줄
        if self._is_isolated_text(text_info, all_texts):
            return True
        
        return False

    def _is_isolated_text(self, text_info: Dict, all_texts: List[Dict]) -> bool:
        """텍스트가 독립된 줄인지 확인"""
        bbox = self._normalize_bbox(text_info.get('bbox', []))
        if len(bbox) < 4:
            return False

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

        margin = 5
        return (inner[0] >= outer[0] - margin and 
                inner[1] >= outer[1] - margin and 
                inner[2] <= outer[2] + margin and 
                inner[3] <= outer[3] + margin)
    
    # analyze_pdf 메서드도 동일한 개선사항 적용
    def analyze_pdf(self, pdf_path: str, 
                   chunk_size: int = 10,
                   progress_callback: Callable = None,
                   frontend_format: bool = False) -> Dict[str, Any]:
        """PDF 문서 전체 분석 - analyze_pdf_with_callbacks 호출"""
        return self.analyze_pdf_with_callbacks(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            page_callback=None,
            frontend_format=frontend_format
        )