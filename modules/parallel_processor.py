import concurrent.futures
from typing import List, Dict, Callable, Any
import multiprocessing
import psutil
import logging
from tqdm import tqdm
import queue
import threading
import time

logger = logging.getLogger(__name__)

class ParallelProcessor:
    def __init__(self, max_workers: int = None):
        """병렬 처리기 초기화
        
        Args:
            max_workers: 최대 워커 수 (None이면 CPU 코어 수 사용)
        """
        if max_workers is None:
            # GPU 사용 시 메모리를 고려하여 워커 수 제한
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = min(cpu_count, 4)  # 최대 4개로 제한
        else:
            self.max_workers = max_workers
        
        # 시스템 리소스 모니터링
        self.memory_threshold = 80  # 메모리 사용률 임계값 (%)
        
    def process_pages_parallel(self, 
                             pages: List[Tuple[int, str]], 
                             process_func: Callable,
                             batch_size: int = 1) -> List[Dict]:
        """페이지를 병렬로 처리
        
        Args:
            pages: (page_number, image_path) 튜플 리스트
            process_func: 각 페이지를 처리할 함수
            batch_size: 배치 크기
            
        Returns:
            처리 결과 리스트
        """
        results = []
        
        # 배치 생성
        batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
        
        # ThreadPoolExecutor 사용 (GPU 작업에 적합)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 진행 상황 표시
            with tqdm(total=len(pages), desc="Processing pages") as pbar:
                # Future 객체 생성
                futures = []
                for batch in batches:
                    # 메모리 체크
                    self._wait_for_memory()
                    
                    future = executor.submit(self._process_batch, batch, process_func)
                    futures.append((batch, future))
                
                # 결과 수집
                for batch, future in futures:
                    try:
                        batch_results = future.result(timeout=300)  # 5분 타임아웃
                        results.extend(batch_results)
                        pbar.update(len(batch))
                    except concurrent.futures.TimeoutError:
                        logger.error(f"배치 처리 타임아웃: {[p[0] for p in batch]}")
                        # 실패한 페이지에 대한 기본 결과 추가
                        for page_num, _ in batch:
                            results.append({
                                'page_index': page_num,
                                'error': 'Processing timeout'
                            })
                        pbar.update(len(batch))
                    except Exception as e:
                        logger.error(f"배치 처리 실패: {str(e)}")
                        for page_num, _ in batch:
                            results.append({
                                'page_index': page_num,
                                'error': str(e)
                            })
                        pbar.update(len(batch))
        
        # 페이지 번호로 정렬
        results.sort(key=lambda x: x.get('page_index', 0))
        return results
    
    def _process_batch(self, batch: List[Tuple[int, str]], 
                      process_func: Callable) -> List[Dict]:
        """배치 처리"""
        batch_results = []
        
        for page_num, image_path in batch:
            try:
                result = process_func(image_path, page_num)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"페이지 {page_num} 처리 실패: {str(e)}")
                batch_results.append({
                    'page_index': page_num,
                    'error': str(e)
                })
        
        return batch_results
    
    def _wait_for_memory(self):
        """메모리 사용률 체크 및 대기"""
        while psutil.virtual_memory().percent > self.memory_threshold:
            logger.warning(f"메모리 사용률이 {self.memory_threshold}%를 초과. 대기 중...")
            time.sleep(5)
    
    def process_with_progress_callback(self,
                                     items: List[Any],
                                     process_func: Callable,
                                     progress_callback: Callable = None) -> List[Dict]:
        """진행 상황 콜백과 함께 처리
        
        Args:
            items: 처리할 항목 리스트
            process_func: 처리 함수
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            처리 결과 리스트
        """
        results = []
        completed = 0
        total = len(items)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_func, item): item for item in items}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total)
                        
                except Exception as e:
                    logger.error(f"처리 실패: {str(e)}")
                    results.append({'error': str(e)})
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total)
        
        return results