import os

class Config:
    # PaddleOCR 설정
    USE_GPU = True
    GPU_MEM = 500
    
    # 레이아웃 감지 모델
    LAYOUT_MODEL = "PP-DocLayout_plus-L"
    
    # OCR 모델
    DET_MODEL = "PP-OCRv5_server_det"
    REC_MODEL = "PP-OCRv5_server_rec"
    
    # BERT 모델 (Figure 참조 분류용)
    BERT_MODEL = "bert-base-uncased"
    
    # API 설정
    API_HOST = "0.0.0.0"
    API_PORT = 12321
    
    # 파일 업로드 설정
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Figure 참조 패턴
    FIGURE_PATTERNS = [
        r'Fig\.\s*\d+',
        r'Figure\s*\d+',
        r'fig\.\s*\d+',
        r'figure\s*\d+',
        r'그림\s*\d+',
        r'도표\s*\d+'
    ]

config = Config()