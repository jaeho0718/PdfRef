from typing import List, Dict, Any, Tuple, Callable
import os
import tempfile
import shutil
import logging
from datetime import datetime

from layout_detector import LayoutDetector
from text_detector import TextDetector
from figure_classifier import FigureClassifier
from figure_mapper import FigureMapper
from pdf_processor import PDFProcessor
from parallel_processor import ParallelProcessor

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

    def process_pages_with_callbacks(self,
                                     pages: List[Tuple[int, str]],
                                     process_func: Callable,
                                     progress_callback: Callable = None,
                                     page_callback: Callable = None) -> List[Dict]:
        """콜백과 함께 페이지 병렬 처리

        Args:
            pages: (page_number, image_path) 튜플 리스트
            process_func: 각 페이지를 처리할 함수
            progress_callback: 진행 상황 콜백
            page_callback: 페이지 완료 콜백

        Returns:
            처리 결과 리스트
        """
        results = []
        completed = 0
        total = len(pages)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Future 객체 생성
            futures = {
                executor.submit(process_func, img_path, page_num): (page_num, img_path)
                for page_num, img_path in pages
            }

            # 완료된 작업 처리
            for future in concurrent.futures.as_completed(futures):
                page_num, img_path = futures[future]

                try:
                    result = future.result(timeout=300)
                    results.append(result)
                    completed += 1

                    # 콜백 호출
                    if progress_callback:
                        progress_callback(completed, total)

                    if page_callback:
                        page_callback(result)

                except concurrent.futures.TimeoutError:
                    logger.error(f"페이지 {page_num} 처리 타임아웃")
                    error_result = {
                        'page_index': page_num,
                        'error': 'Processing timeout'
                    }
                    results.append(error_result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total)

                except Exception as e:
                    logger.error(f"페이지 {page_num} 처리 실패: {str(e)}")
                    error_result = {
                        'page_index': page_num,
                        'error': str(e)
                    }
                    results.append(error_result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total)

                # 이미지 파일 정리
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                except:
                    pass

        # 페이지 번호로 정렬
        results.sort(key=lambda x: x.get('page_index', 0))
        return results
        
    def analyze_pdf(self, pdf_path: str, 
                   chunk_size: int = 10,
                   progress_callback: Callable = None) -> Dict[str, Any]:
        """PDF 문서 전체 분석
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 병렬 처리 청크 크기
            progress_callback: 진행 상황 콜백
            
        Returns:
            분석 결과
        """
        start_time = datetime.now()
        temp_dir = tempfile.mkdtemp()
        
        try:
            # PDF 정보 추출
            pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
            total_pages = pdf_info['total_pages']
            
            logger.info(f"PDF 분석 시작: {total_pages} 페이지")
            
            # PDF를 이미지로 변환 (Generator 사용)
            pages = list(self.pdf_processor.convert_pdf_to_images(pdf_path))
            
            # 병렬 처리
            results = self.parallel_processor.process_pages_parallel(
                pages,
                self._analyze_single_page,
                batch_size=1
            )
            
            # 전체 문서에 대한 Figure 맵핑 최적화
            all_figure_refs = []
            all_figure_layouts = []
            
            for result in results:
                if 'error' not in result:
                    all_figure_refs.extend(result.get('figure_references', []))
                    all_figure_layouts.extend(result.get('figure_layouts', []))
            
            # Cross-page Figure 맵핑
            optimized_mappings = self._optimize_figure_mappings(
                all_figure_refs, 
                all_figure_layouts
            )
            
            # 결과에 최적화된 맵핑 적용
            for result in results:
                if 'error' not in result and 'figure_references' in result:
                    page_idx = result['page_index']
                    result['figure_references'] = [
                        ref for ref in optimized_mappings 
                        if ref.get('page_index') == page_idx
                    ]
            
            # 최종 결과 구성
            analysis_result = {
                'status': 'success',
                'pdf_info': pdf_info,
                'total_pages': total_pages,
                'pages': results,
                'summary': self._generate_summary(results),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PDF 분석 실패: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        finally:
            # 임시 디렉토리 정리
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _analyze_single_page(self, image_path: str, page_index: int) -> Dict[str, Any]:
        """단일 페이지 분석"""
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
            
            # 4. Figure 참조 추출
            figure_references = self.figure_classifier.extract_figure_references(
                recognized_texts
            )
            
            # 5. 페이지 내 Figure 맵핑 (초기 맵핑)
            mapped_references = self.figure_mapper.map_references_to_figures(
                figure_references,
                layout_result['figure_layouts'],
                page_index
            )
            
            # 페이지 인덱스 추가
            for ref in mapped_references:
                ref['page_index'] = page_index
            
            return {
                'page_index': page_index,
                'layouts': layout_result['layouts'],
                'figure_layouts': layout_result['figure_layouts'],
                'recognized_texts': recognized_texts,
                'figure_references': mapped_references,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"페이지 {page_index} 분석 실패: {str(e)}")
            return {
                'page_index': page_index,
                'status': 'error',
                'error': str(e)
            }
    
    def _optimize_figure_mappings(self, 
                                all_refs: List[Dict], 
                                all_figures: List[Dict]) -> List[Dict]:
        """전체 문서에 대한 Figure 맵핑 최적화"""
        # Figure 번호별로 그룹화
        ref_groups = {}
        for ref in all_refs:
            fig_num = ref.get('figure_number')
            if fig_num not in ref_groups:
                ref_groups[fig_num] = []
            ref_groups[fig_num].append(ref)
        
        # Figure 레이아웃도 번호별로 그룹화 (휴리스틱)
        figure_groups = {}
        for idx, fig in enumerate(all_figures):
            # 페이지와 순서를 기반으로 추정
            estimated_num = idx + 1  # 간단한 휴리스틱
            figure_groups[estimated_num] = fig
        
        # 최적화된 맵핑
        optimized_refs = []
        for fig_num, refs in ref_groups.items():
            if fig_num in figure_groups:
                target_figure = figure_groups[fig_num]
                for ref in refs:
                    ref['mapped_figure_id'] = target_figure['figure_id']
                    ref['mapped_figure_bbox'] = target_figure['bbox']
                    ref['mapped_figure_page'] = target_figure['page_index']
                    optimized_refs.append(ref)
            else:
                # 맵핑 실패한 경우
                for ref in refs:
                    ref['mapped_figure_id'] = None
                    ref['mapped_figure_bbox'] = None
                    ref['mapped_figure_page'] = None
                    optimized_refs.append(ref)
        
        return optimized_refs
    
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

    def analyze_pdf_with_callbacks(self, 
                                  pdf_path: str,
                                  chunk_size: int = 10,
                                  progress_callback: Callable = None,
                                  page_callback: Callable = None) -> Dict[str, Any]:
        """콜백과 함께 PDF 분석 (CLI용)
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 병렬 처리 청크 크기
            progress_callback: 진행 상황 콜백 (current, total)
            page_callback: 페이지 완료 콜백 (page_result)
            
        Returns:
            분석 결과
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
                self._analyze_single_page,
                progress_callback=progress_callback,
                page_callback=page_callback
            )
            
            # Figure 맵핑 최적화
            all_figure_refs = []
            all_figure_layouts = []
            
            for result in results:
                if 'error' not in result:
                    all_figure_refs.extend(result.get('figure_references', []))
                    all_figure_layouts.extend(result.get('figure_layouts', []))
            
            # Cross-page Figure 맵핑
            optimized_mappings = self._optimize_figure_mappings(
                all_figure_refs, 
                all_figure_layouts
            )
            
            # 결과에 최적화된 맵핑 적용
            for result in results:
                if 'error' not in result and 'figure_references' in result:
                    page_idx = result['page_index']
                    result['figure_references'] = [
                        ref for ref in optimized_mappings 
                        if ref.get('page_index') == page_idx
                    ]
            
            # 최종 결과 구성
            analysis_result = {
                'status': 'success',
                'pdf_info': pdf_info,
                'total_pages': total_pages,
                'pages': results,
                'summary': self._generate_summary(results),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PDF 분석 실패: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        finally:
            # 임시 디렉토리 정리
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)