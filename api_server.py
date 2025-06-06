from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from typing import Dict, Any
import json
import redis
from celery import Celery
from datetime import datetime
import logging

from modules.document_analyzer import DocumentAnalyzer
from config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 설정
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Redis 설정
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Celery 설정
celery = Celery(
    'pdf_analyzer',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# 업로드 폴더 생성
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs('results', exist_ok=True)

# 문서 분석기 초기화
document_analyzer = DocumentAnalyzer()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@celery.task(bind=True)
def analyze_pdf_task(self, task_id: str, pdf_path: str):
    """비동기 PDF 분석 태스크"""
    try:
        # 진행 상황 업데이트 콜백
        def update_progress(completed, total):
            progress = (completed / total) * 100
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': completed,
                    'total': total,
                    'progress': progress
                }
            )
            # Redis에도 저장
            redis_client.hset(
                f"task:{task_id}",
                mapping={
                    'progress': progress,
                    'current': completed,
                    'total': total,
                    'status': 'processing'
                }
            )
        
        # PDF 분석 수행
        result = document_analyzer.analyze_pdf(
            pdf_path,
            chunk_size=10,
            progress_callback=update_progress
        )
        
        # 결과 저장
        result_path = f"results/{task_id}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Redis에 완료 상태 저장
        redis_client.hset(
            f"task:{task_id}",
            mapping={
                'status': 'completed',
                'result_path': result_path,
                'completed_at': datetime.now().isoformat()
            }
        )
        
        # 임시 파일 삭제
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        return {
            'task_id': task_id,
            'status': 'completed',
            'result_path': result_path
        }
        
    except Exception as e:
        logger.error(f"PDF 분석 실패: {str(e)}")
        # 에러 상태 저장
        redis_client.hset(
            f"task:{task_id}",
            mapping={
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
        )
        
        # 임시 파일 삭제
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'message': 'Figure Reference System is running',
        'redis': redis_client.ping()
    })

@app.route('/analyze/pdf', methods=['POST'])
def analyze_pdf():
    """PDF 분석 요청 (비동기)"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename) or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # 파일 저장
        filename = secure_filename(file.filename)
        task_id = str(uuid.uuid4())
        unique_filename = f"{task_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # 비동기 태스크 시작
        task = analyze_pdf_task.apply_async(args=[task_id, filepath])
        
        # Redis에 태스크 정보 저장
        redis_client.hset(
            f"task:{task_id}",
            mapping={
                'task_id': task_id,
                'celery_task_id': task.id,
                'filename': filename,
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }
        )
        
        return jsonify({
            'status': 'accepted',
            'task_id': task_id,
            'message': 'PDF analysis started',
            'check_status_url': f'/analyze/status/{task_id}'
        }), 202
        
    except Exception as e:
        logger.error(f"PDF 분석 요청 실패: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/analyze/status/<task_id>', methods=['GET'])
def check_analysis_status(task_id):
    """분석 상태 확인"""
    try:
        # Redis에서 태스크 정보 조회
        task_info = redis_client.hgetall(f"task:{task_id}")
        
        if not task_info:
            return jsonify({'error': 'Task not found'}), 404
        
        response = {
            'task_id': task_id,
            'status': task_info.get('status', 'unknown'),
            'filename': task_info.get('filename'),
            'created_at': task_info.get('created_at')
        }
        
        # 진행 중인 경우 진행률 추가
        if task_info.get('status') == 'processing':
            response['progress'] = float(task_info.get('progress', 0))
            response['current_page'] = int(task_info.get('current', 0))
            response['total_pages'] = int(task_info.get('total', 0))
        
        # 완료된 경우 결과 URL 추가
        elif task_info.get('status') == 'completed':
            response['result_url'] = f'/analyze/result/{task_id}'
            response['completed_at'] = task_info.get('completed_at')
        
        # 실패한 경우 에러 메시지 추가
        elif task_info.get('status') == 'failed':
            response['error'] = task_info.get('error')
            response['failed_at'] = task_info.get('failed_at')
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"상태 확인 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/result/<task_id>', methods=['GET'])
def get_analysis_result(task_id):
    """분석 결과 조회"""
    try:
        # Redis에서 태스크 정보 확인
        task_info = redis_client.hgetall(f"task:{task_id}")
        
        if not task_info:
            return jsonify({'error': 'Task not found'}), 404
        
        if task_info.get('status') != 'completed':
            return jsonify({
                'error': 'Analysis not completed',
                'status': task_info.get('status')
            }), 400
        
        # 결과 파일 읽기
        result_path = task_info.get('result_path')
        if not result_path or not os.path.exists(result_path):
            return jsonify({'error': 'Result file not found'}), 404
        
        # 결과 크기 확인
        file_size = os.path.getsize(result_path)
        
        # 작은 파일은 JSON으로 반환
        if file_size < 10 * 1024 * 1024:  # 10MB 미만
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return jsonify(result), 200
        else:
            # 큰 파일은 다운로드로 제공
            return send_file(
                result_path,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'analysis_result_{task_id}.json'
            )
        
    except Exception as e:
        logger.error(f"결과 조회 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/result/<task_id>/page/<int:page_index>', methods=['GET'])
def get_page_result(task_id, page_index):
    """특정 페이지 결과만 조회"""
    try:
        # Redis에서 태스크 정보 확인
        task_info = redis_client.hgetall(f"task:{task_id}")
        
        if not task_info:
            return jsonify({'error': 'Task not found'}), 404
        
        if task_info.get('status') != 'completed':
            return jsonify({
                'error': 'Analysis not completed',
                'status': task_info.get('status')
            }), 400
        
        # 결과 파일 읽기
        result_path = task_info.get('result_path')
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 특정 페이지 결과 찾기
        pages = result.get('pages', [])
        page_result = None
        
        for page in pages:
            if page.get('page_index') == page_index:
                page_result = page
                break
        
        if page_result is None:
            return jsonify({'error': 'Page not found'}), 404
        
        return jsonify({
            'task_id': task_id,
            'page_index': page_index,
            'page_data': page_result,
            'total_pages': result.get('total_pages', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"페이지 결과 조회 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/cancel/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    """분석 취소"""
    try:
        # Redis에서 태스크 정보 조회
        task_info = redis_client.hgetall(f"task:{task_id}")
        
        if not task_info:
            return jsonify({'error': 'Task not found'}), 404
        
        # Celery 태스크 취소
        celery_task_id = task_info.get('celery_task_id')
        if celery_task_id:
            celery.control.revoke(celery_task_id, terminate=True)
        
        # Redis 상태 업데이트
        redis_client.hset(
            f"task:{task_id}",
            mapping={
                'status': 'cancelled',
                'cancelled_at': datetime.now().isoformat()
            }
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis cancelled',
            'task_id': task_id
        }), 200
        
    except Exception as e:
        logger.error(f"분석 취소 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)