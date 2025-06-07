from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_restx import Api, Resource, fields, Namespace
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from typing import Dict, Any, Generator
import json
import redis
from celery import Celery
from datetime import datetime
import logging
import time
import ijson

from modules.document_analyzer import DocumentAnalyzer
from config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 앱 설정
app = Flask(__name__)
CORS(app)

# Swagger 설정
api = Api(
    app,
    version='1.0',
    title='PDF Figure Reference Analyzer API',
    description='PDF 문서에서 Figure 참조를 분석하는 API',
    doc='/docs'
)

# 네임스페이스 설정
ns_analyze = api.namespace('analyze', description='문서 분석 작업')
ns_status = api.namespace('status', description='작업 상태 확인')
ns_result = api.namespace('result', description='분석 결과 조회')

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

# Swagger 모델 정의
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True, help='PDF 파일')

task_response = api.model('TaskResponse', {
    'status': fields.String(description='작업 상태'),
    'task_id': fields.String(description='작업 ID'),
    'message': fields.String(description='메시지'),
    'check_status_url': fields.String(description='상태 확인 URL'),
    'stream_url': fields.String(description='진행 상황 스트림 URL')
})

status_response = api.model('StatusResponse', {
    'task_id': fields.String(description='작업 ID'),
    'status': fields.String(description='작업 상태', enum=['pending', 'processing', 'completed', 'failed', 'cancelled']),
    'filename': fields.String(description='파일명'),
    'created_at': fields.String(description='생성 시간'),
    'progress': fields.Float(description='진행률 (0-100)'),
    'current_page': fields.Integer(description='현재 처리 중인 페이지'),
    'total_pages': fields.Integer(description='전체 페이지 수'),
    'result_url': fields.String(description='결과 조회 URL'),
    'completed_at': fields.String(description='완료 시간'),
    'error': fields.String(description='에러 메시지')
})

error_response = api.model('ErrorResponse', {
    'error': fields.String(description='에러 메시지'),
    'status': fields.String(description='상태 코드')
})

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
                    'status': 'processing',
                    'last_update': datetime.now().isoformat()
                }
            )
            
            # 진행 상황 이벤트 발행
            event_data = {
                'task_id': task_id,
                'progress': progress,
                'current': completed,
                'total': total,
                'timestamp': datetime.now().isoformat()
            }
            redis_client.publish(f"progress:{task_id}", json.dumps(event_data))
        
        # 페이지별 결과 콜백
        def page_callback(page_result):
            # 페이지 결과를 Redis에 저장 (스트리밍용)
            page_idx = page_result.get('page_index', 0)
            redis_client.hset(
                f"task:{task_id}:pages",
                f"page_{page_idx}",
                json.dumps(page_result)
            )
            
            # 페이지 완료 이벤트 발행
            event_data = {
                'type': 'page_completed',
                'task_id': task_id,
                'page_index': page_idx,
                'timestamp': datetime.now().isoformat()
            }
            redis_client.publish(f"progress:{task_id}", json.dumps(event_data))
        
        # PDF 분석 수행
        result = document_analyzer.analyze_pdf_with_callbacks(
            pdf_path,
            chunk_size=10,
            progress_callback=update_progress,
            page_callback=page_callback
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
                'completed_at': datetime.now().isoformat(),
                'progress': 100
            }
        )
        
        # 완료 이벤트 발행
        event_data = {
            'type': 'completed',
            'task_id': task_id,
            'result_path': result_path,
            'timestamp': datetime.now().isoformat()
        }
        redis_client.publish(f"progress:{task_id}", json.dumps(event_data))
        
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
        
        # 에러 이벤트 발행
        event_data = {
            'type': 'error',
            'task_id': task_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        redis_client.publish(f"progress:{task_id}", json.dumps(event_data))
        
        # 임시 파일 삭제
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        raise

def generate_progress_stream(task_id: str) -> Generator[str, None, None]:
    """진행 상황 SSE 스트림 생성"""
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"progress:{task_id}")
    
    # 초기 상태 전송
    task_info = redis_client.hgetall(f"task:{task_id}")
    if task_info:
        yield f"data: {json.dumps(task_info)}\n\n"
    
    # 실시간 업데이트 스트리밍
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                yield f"data: {message['data']}\n\n"
                
                # 완료 또는 에러 시 스트림 종료
                data = json.loads(message['data'])
                if data.get('type') in ['completed', 'error']:
                    break
    finally:
        pubsub.unsubscribe()
        pubsub.close()

@ns_analyze.route('/pdf')
class PDFAnalysis(Resource):
    @ns_analyze.expect(upload_parser)
    @ns_analyze.marshal_with(task_response, code=202)
    @ns_analyze.response(400, 'Bad Request', error_response)
    @ns_analyze.response(500, 'Internal Server Error', error_response)
    def post(self):
        """PDF 문서 분석 요청"""
        try:
            # 파일 확인
            if 'file' not in request.files:
                return {'error': 'No file provided', 'status': 'error'}, 400
            
            file = request.files['file']
            if file.filename == '':
                return {'error': 'No file selected', 'status': 'error'}, 400
            
            if not allowed_file(file.filename) or not file.filename.lower().endswith('.pdf'):
                return {'error': 'Only PDF files are allowed', 'status': 'error'}, 400
            
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
            
            return {
                'status': 'accepted',
                'task_id': task_id,
                'message': 'PDF analysis started',
                'check_status_url': f'/status/{task_id}',
                'stream_url': f'/analyze/stream/{task_id}'
            }, 202
            
        except Exception as e:
            logger.error(f"PDF 분석 요청 실패: {str(e)}")
            return {'error': str(e), 'status': 'error'}, 500

@ns_analyze.route('/stream/<string:task_id>')
class ProgressStream(Resource):
    @ns_analyze.response(200, 'Success')
    @ns_analyze.response(404, 'Task not found', error_response)
    def get(self, task_id):
        """분석 진행 상황 실시간 스트림 (SSE)"""
        # 태스크 존재 확인
        task_info = redis_client.hgetall(f"task:{task_id}")
        if not task_info:
            return {'error': 'Task not found', 'status': 'error'}, 404
        
        return Response(
            stream_with_context(generate_progress_stream(task_id)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

@ns_status.route('/<string:task_id>')
class TaskStatus(Resource):
    @ns_status.marshal_with(status_response)
    @ns_status.response(404, 'Task not found', error_response)
    def get(self, task_id):
        """작업 상태 확인"""
        try:
            # Redis에서 태스크 정보 조회
            task_info = redis_client.hgetall(f"task:{task_id}")
            
            if not task_info:
                return {'error': 'Task not found', 'status': 'error'}, 404
            
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
                response['result_url'] = f'/result/{task_id}'
                response['completed_at'] = task_info.get('completed_at')
            
            # 실패한 경우 에러 메시지 추가
            elif task_info.get('status') == 'failed':
                response['error'] = task_info.get('error')
            
            return response, 200
            
        except Exception as e:
            logger.error(f"상태 확인 실패: {str(e)}")
            return {'error': str(e), 'status': 'error'}, 500

@ns_result.route('/<string:task_id>')
class AnalysisResult(Resource):
    @ns_result.response(200, 'Success')
    @ns_result.response(400, 'Bad Request', error_response)
    @ns_result.response(404, 'Not found', error_response)
    def get(self, task_id):
        """분석 결과 조회"""
        try:
            # Redis에서 태스크 정보 확인
            task_info = redis_client.hgetall(f"task:{task_id}")
            
            if not task_info:
                return {'error': 'Task not found', 'status': 'error'}, 404
            
            if task_info.get('status') != 'completed':
                return {
                    'error': 'Analysis not completed',
                    'status': task_info.get('status')
                }, 400
            
            # 결과 파일 경로
            result_path = task_info.get('result_path')
            if not result_path or not os.path.exists(result_path):
                return {'error': 'Result file not found', 'status': 'error'}, 404
            
            # 스트리밍 옵션 확인
            stream = request.args.get('stream', 'false').lower() == 'true'
            
            if stream:
                # 대용량 JSON 스트리밍
                def generate():
                    with open(result_path, 'rb') as f:
                        parser = ijson.parse(f)
                        for prefix, event, value in parser:
                            if event == 'start_map':
                                yield '{'
                            elif event == 'map_key':
                                yield f'"{value}":'
                            elif event == 'string':
                                yield f'"{value}"'
                            elif event == 'number':
                                yield str(value)
                            elif event == 'start_array':
                                yield '['
                            elif event == 'end_array':
                                yield ']'
                            elif event == 'end_map':
                                yield '}'
                            # 더 많은 이벤트 처리...
                
                return Response(
                    stream_with_context(generate()),
                    mimetype='application/json',
                    headers={
                        'Content-Disposition': f'attachment; filename=result_{task_id}.json'
                    }
                )
            else:
                # 일반 응답
                file_size = os.path.getsize(result_path)
                
                # 작은 파일은 직접 반환
                if file_size < 10 * 1024 * 1024:  # 10MB 미만
                    with open(result_path, 'r', encoding='utf-8') as f:
                        return json.load(f), 200
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
            return {'error': str(e), 'status': 'error'}, 500

@ns_result.route('/<string:task_id>/stream')
class StreamedResult(Resource):
    @ns_result.response(200, 'Success')
    @ns_result.response(404, 'Not found', error_response)
    def get(self, task_id):
        """대용량 분석 결과 스트리밍"""
        try:
            # Redis에서 태스크 정보 확인
            task_info = redis_client.hgetall(f"task:{task_id}")
            
            if not task_info:
                return {'error': 'Task not found', 'status': 'error'}, 404
            
            if task_info.get('status') != 'completed':
                return {
                    'error': 'Analysis not completed',
                    'status': task_info.get('status')
                }, 400
            
            result_path = task_info.get('result_path')
            if not result_path or not os.path.exists(result_path):
                return {'error': 'Result file not found', 'status': 'error'}, 404
            
            def generate_json_stream():
                """JSON 파일을 청크 단위로 스트리밍"""
                with open(result_path, 'r', encoding='utf-8') as f:
                    chunk_size = 8192  # 8KB 청크
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            
            return Response(
                stream_with_context(generate_json_stream()),
                mimetype='application/json',
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Cache-Control': 'no-cache',
                    'X-Content-Type-Options': 'nosniff'
                }
            )
            
        except Exception as e:
            logger.error(f"스트리밍 실패: {str(e)}")
            return {'error': str(e), 'status': 'error'}, 500

@ns_result.route('/<string:task_id>/page/<int:page_index>')
class PageResult(Resource):
    @ns_result.response(200, 'Success')
    @ns_result.response(404, 'Not found', error_response)
    def get(self, task_id, page_index):
        """특정 페이지 결과 조회"""
        try:
            # Redis에서 페이지 데이터 조회
            page_data = redis_client.hget(f"task:{task_id}:pages", f"page_{page_index}")
            
            if page_data:
                return json.loads(page_data), 200
            
            # Redis에 없으면 파일에서 조회
            task_info = redis_client.hgetall(f"task:{task_id}")
            if not task_info:
                return {'error': 'Task not found', 'status': 'error'}, 404
            
            result_path = task_info.get('result_path')
            if not result_path or not os.path.exists(result_path):
                return {'error': 'Result file not found', 'status': 'error'}, 404
            
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # 특정 페이지 찾기
            pages = result.get('pages', [])
            for page in pages:
                if page.get('page_index') == page_index:
                    return page, 200
            
            return {'error': 'Page not found', 'status': 'error'}, 404
            
        except Exception as e:
            logger.error(f"페이지 결과 조회 실패: {str(e)}")
            return {'error': str(e), 'status': 'error'}, 500

@ns_analyze.route('/cancel/<string:task_id>')
class CancelAnalysis(Resource):
    @ns_analyze.response(200, 'Success')
    @ns_analyze.response(404, 'Not found', error_response)
    def post(self, task_id):
        """분석 작업 취소"""
        try:
            # Redis에서 태스크 정보 조회
            task_info = redis_client.hgetall(f"task:{task_id}")
            
            if not task_info:
                return {'error': 'Task not found', 'status': 'error'}, 404
            
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
            
            # 취소 이벤트 발행
            event_data = {
                'type': 'cancelled',
                'task_id': task_id,
                'timestamp': datetime.now().isoformat()
            }
            redis_client.publish(f"progress:{task_id}", json.dumps(event_data))
            
            return {
                'status': 'success',
                'message': 'Analysis cancelled',
                'task_id': task_id
            }, 200
            
        except Exception as e:
            logger.error(f"분석 취소 실패: {str(e)}")
            return {'error': str(e), 'status': 'error'}, 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """시스템 상태 확인"""
    try:
        # Redis 연결 확인
        redis_status = redis_client.ping()
        
        return jsonify({
            'status': 'healthy',
            'message': 'Figure Reference System is running',
            'redis': redis_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)