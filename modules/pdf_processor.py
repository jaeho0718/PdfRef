import os
import tempfile
from typing import List, Dict, Tuple, Generator
from pdf2image import convert_from_path
import PyPDF2
from PIL import Image
import uuid
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, dpi: int = 200, output_format: str = 'JPEG'):
        """PDF 프로세서 초기화
        
        Args:
            dpi: 이미지 변환 해상도
            output_format: 출력 이미지 포맷
        """
        self.dpi = dpi
        self.output_format = output_format
        
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """PDF 정보 추출"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return {
                    'total_pages': len(pdf_reader.pages),
                    'encrypted': pdf_reader.is_encrypted,
                    'metadata': self._extract_metadata(pdf_reader)
                }
        except Exception as e:
            logger.error(f"PDF 정보 추출 실패: {str(e)}")
            return {
                'total_pages': 0,
                'encrypted': False,
                'metadata': {},
                'error': str(e)
            }
    
    def _extract_metadata(self, pdf_reader) -> Dict:
        """PDF 메타데이터 추출"""
        metadata = {}
        if pdf_reader.metadata:
            for key, value in pdf_reader.metadata.items():
                if isinstance(value, str):
                    metadata[key] = value
        return metadata
    
    def convert_pdf_to_images(self, pdf_path: str, 
                            start_page: int = None, 
                            end_page: int = None) -> Generator[Tuple[int, str], None, None]:
        """PDF를 이미지로 변환 (Generator 사용)
        
        Args:
            pdf_path: PDF 파일 경로
            start_page: 시작 페이지 (1-based)
            end_page: 종료 페이지 (1-based)
            
        Yields:
            (page_number, image_path, width, height) 튜플
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 페이지 범위 설정
            first_page = start_page if start_page else 1
            last_page = end_page if end_page else None
            
            # PDF를 이미지로 변환
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.output_format.lower(),
                first_page=first_page,
                last_page=last_page,
                thread_count=4  # 변환 시 병렬 처리
            )
            
            # 각 이미지 저장 및 경로 반환
            for idx, image in enumerate(images):
                # 이미지 크기 정보 추출
                width, height = image.size
                
                page_num = (first_page - 1) + idx
                image_filename = f"page_{page_num}_{uuid.uuid4()}.{self.output_format.lower()}"
                image_path = os.path.join(temp_dir, image_filename)
                
                image.save(image_path, self.output_format)
                yield (page_num, image_path, width, height)
                
        except Exception as e:
            logger.error(f"PDF 변환 실패: {str(e)}")
            raise
        finally:
            # 임시 디렉토리는 호출자가 정리하도록 함
            pass
    
    def split_pdf_for_parallel_processing(self, pdf_path: str, 
                                        chunk_size: int = 10) -> List[Dict]:
        """병렬 처리를 위해 PDF를 청크로 분할
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 각 청크의 페이지 수
            
        Returns:
            청크 정보 리스트
        """
        pdf_info = self.get_pdf_info(pdf_path)
        total_pages = pdf_info['total_pages']
        
        chunks = []
        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            chunks.append({
                'chunk_id': len(chunks),
                'start_page': start + 1,  # 1-based
                'end_page': end,
                'total_pages': end - start
            })
        
        return chunks