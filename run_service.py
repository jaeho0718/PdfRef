import subprocess
import os
import sys
import time
import signal

def check_redis():
    """Redis 실행 확인"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True
    except:
        return False

def start_redis_windows():
    """Windows에서 Redis 시작"""
    if check_redis():
        print("✓ Redis가 이미 실행 중입니다")
        return
    
    print("Redis 시작 중...")
    
    # Windows용 Redis 실행 시도
    redis_paths = [
        r"C:\Program Files\Redis\redis-server.exe",
        r"C:\Redis\redis-server.exe",
        "redis-server.exe"
    ]
    
    for redis_path in redis_paths:
        if os.path.exists(redis_path):
            subprocess.Popen([redis_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
            time.sleep(3)
            if check_redis():
                print("✓ Redis가 시작되었습니다")
                return
    
    # WSL로 Redis 시작 시도
    try:
        subprocess.Popen(["wsl", "redis-server"], 
                       creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(3)
        if check_redis():
            print("✓ Redis가 WSL에서 시작되었습니다")
            return
    except:
        pass
    
    print("⚠ Redis를 시작할 수 없습니다. 수동으로 시작해주세요.")

def main():
    print("="*50)
    print("PDF Figure Reference Analyzer 서버 시작")
    print("="*50)
    
    # 1. 필요한 디렉토리 생성
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 2. Redis 시작
    print("\n[1/3] Redis 서버 확인...")
    start_redis_windows()
    
    # 3. Celery 워커 시작
    print("\n[2/3] Celery 워커 시작...")
    celery_cmd = [
        sys.executable,
        "-m", "celery",
        "-A", "api_server.celery",
        "worker",
        "--loglevel=info",
        "--pool=threads",  # Windows에서는 threads 사용
        "--concurrency=2"
    ]
    
    celery_process = subprocess.Popen(
        celery_cmd,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(3)
    print("✓ Celery 워커가 새 창에서 시작되었습니다")
    
    # 4. Flask 서버 시작
    print("\n[3/3] Flask API 서버 시작...")
    print(f"\n서버가 포트 12322에서 실행됩니다")
    print("\n접속 URL:")
    print("- 로컬: http://localhost:12322/docs")
    print("- 상태 확인: http://localhost:12322/health")
    print("- ngrok URL: https://[your-ngrok-id].ngrok.io/docs")
    print("\n종료하려면 Ctrl+C를 누르세요\n")
    
    # Flask 서버 실행
    try:
        flask_cmd = [sys.executable, "api_server.py"]
        subprocess.call(flask_cmd)
    except KeyboardInterrupt:
        print("\n\n서버를 종료합니다...")
        # Celery 프로세스 종료
        try:
            celery_process.terminate()
        except:
            pass

if __name__ == "__main__":
    main()