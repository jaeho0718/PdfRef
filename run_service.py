import subprocess
import os
import sys
import time

def start_redis():
    """Redis 서버 시작"""
    print("Starting Redis server...")
    subprocess.Popen(['redis-server'], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    time.sleep(2)
    print("Redis server started")

def start_celery():
    """Celery 워커 시작"""
    print("Starting Celery worker...")
    celery_cmd = [
        'celery', '-A', 'api_server.celery', 'worker',
        '--loglevel=info',
        '--concurrency=4',
        '--pool=threads'  # GPU 작업을 위해 threads 사용
    ]
    subprocess.Popen(celery_cmd)
    time.sleep(3)
    print("Celery worker started")

def start_flask():
    """Flask 서버 시작"""
    print("Starting Flask server...")
    flask_cmd = ['python', 'api_server.py']
    subprocess.Popen(flask_cmd)
    print("Flask server started")

def main():
    """모든 서비스 시작"""
    try:
        # Redis 확인
        try:
            import redis
            r = redis.Redis()
            r.ping()
            print("Redis is already running")
        except:
            start_redis()
        
        # Celery 시작
        start_celery()
        
        # Flask 시작
        start_flask()
        
        print("\nAll services started successfully!")
        print("API Server: http://localhost:5000")
        print("\nPress Ctrl+C to stop all services")
        
        # 계속 실행
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down services...")
        # 프로세스 종료는 OS가 처리하도록 함
        sys.exit(0)

if __name__ == '__main__':
    main()