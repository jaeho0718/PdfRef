# PDF Figure Reference Analyzer API Documentation

## 개요

PDF Figure Reference Analyzer는 전공책이나 논문에서 'Fig.28'과 같은 Figure 참조를 자동으로 감지하고, 해당 Figure와 매핑하는 서비스입니다. PaddleOCR과 BERT 모델을 활용하여 높은 정확도의 분석을 제공합니다.

### 주요 기능

- **레이아웃 감지**: PaddleOCR PP-DocLayout_plus-L 모델을 사용한 문서 레이아웃 분석
- **텍스트 인식**: PP-OCRv5 모델을 이용한 고정밀 텍스트 추출
- **Figure 분류**: 레이아웃에서 Figure 영역 자동 분류
- **참조 추출**: BERT 모델 기반 Figure 참조 텍스트 감지
- **지능형 매핑**: Figure 참조와 실제 Figure 간의 자동 매핑
- **실시간 처리**: SSE를 통한 실시간 진행 상황 스트리밍

## 시스템 요구사항

- Python 3.8+
- CUDA 11.8 (GPU 가속)
- Redis (비동기 처리)
- PaddlePaddle
- PyTorch + Transformers

## 기본 정보

- **Base URL**: `http://localhost:12322`
- **API 문서**: `http://localhost:12322/docs` (Swagger UI)
- **Content-Type**: `application/json`, `multipart/form-data`
- **Max File Size**: 50MB

## 인증

현재 버전에서는 인증이 필요하지 않습니다.

---

## API 엔드포인트

### 1. PDF 분석 요청

PDF 문서를 업로드하고 비동기 분석을 시작합니다.

```http
POST /analyze/pdf
```

**Parameters:**
- `file` (required): PDF 파일
- `format` (optional): 응답 형식
  - `frontend` (default): 프론트엔드 친화적 형식
  - `raw`: 원본 분석 결과 형식

**Request Example:**
```bash
curl -X POST \
  'http://localhost:12322/analyze/pdf' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@document.pdf' \
  -F 'format=frontend'
```

**Response (202 Accepted):**
```json
{
  "status": "accepted",
  "task_id": "uuid-string",
  "message": "PDF analysis started",
  "check_status_url": "/status/{task_id}",
  "stream_url": "/analyze/stream/{task_id}"
}
```

**Error Responses:**
- `400`: 잘못된 파일 형식 또는 파일 없음
- `413`: 파일 크기 초과 (50MB)
- `500`: 서버 내부 오류

---

### 2. 실시간 진행 상황 스트리밍

분석 진행 상황을 실시간으로 스트리밍합니다. (Server-Sent Events)

```http
GET /analyze/stream/{task_id}
```

**Response (200 OK):**
```
Content-Type: text/event-stream

data: {"task_id": "uuid", "status": "processing", "progress": 25.5, "current": 5, "total": 20, "timestamp": "2024-01-01T12:00:00Z"}

data: {"task_id": "uuid", "status": "processing", "progress": 50.0, "current": 10, "total": 20, "timestamp": "2024-01-01T12:01:00Z"}

data: {"task_id": "uuid", "status": "completed", "progress": 100.0, "current": 20, "total": 20, "timestamp": "2024-01-01T12:02:00Z"}
```

**JavaScript Example:**
```javascript
const eventSource = new EventSource('/analyze/stream/uuid-string');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}% (${data.current}/${data.total})`);
  
  if (data.status === 'completed') {
    eventSource.close();
    // 결과 조회
    fetchResult(data.task_id);
  }
};
```

---

### 3. 작업 상태 확인

분석 작업의 현재 상태를 확인합니다.

```http
GET /status/{task_id}
```

**Response (200 OK):**
```json
{
  "task_id": "uuid-string",
  "status": "processing",
  "filename": "document.pdf",
  "created_at": "2024-01-01T12:00:00Z",
  "progress": 75.5,
  "current_page": 15,
  "total_pages": 20
}
```

**Status Values:**
- `pending`: 대기 중
- `processing`: 처리 중
- `completed`: 완료
- `failed`: 실패
- `cancelled`: 취소됨

**Completed Status Response:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "filename": "document.pdf",
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:05:00Z",
  "result_url": "/result/uuid-string"
}
```

---

### 4. 분석 결과 조회

완료된 분석의 전체 결과를 조회합니다.

```http
GET /result/{task_id}
```

**Query Parameters:**
- `stream` (optional): `true`로 설정 시 대용량 파일 스트리밍

**Response (200 OK) - Frontend Format:**
```json
{
  "title": "Research Paper Title",
  "chapters": [
    {
      "title": "Introduction",
      "chapter_number": 1,
      "start_page": 0,
      "end_page": 5,
      "sections": ["1.1", "1.2"]
    }
  ],
  "pages": [
    {
      "page_index": 0,
      "chapter": "1",
      "section": "1.1",
      "layouts": [
        {
          "type": "text",
          "bounding_box": [100, 150, 500, 200],
          "text": "As shown in Figure 1, the system architecture consists of...",
          "confidence": 0.95
        },
        {
          "type": "figure",
          "figure_id": "ch1_fig1",
          "bounding_box": [100, 300, 500, 600],
          "figure_caption": "Figure 1. System Architecture",
          "related_chapter": 1,
          "figure_number": 1
        },
        {
          "type": "figure_reference",
          "referenced_figure_id": "ch1_fig1",
          "bounding_box": [200, 150, 250, 170],
          "reference_text": "Figure 1",
          "figure_number": 1,
          "confidence": 0.92
        }
      ],
      "full_text": "Introduction text content..."
    }
  ],
  "figures": [
    {
      "figure_id": "ch1_fig1",
      "figure_number": 1,
      "bounding_box": [100, 300, 500, 600],
      "page_index": 0,
      "chapter": 1,
      "section": null,
      "caption": "Figure 1. System Architecture",
      "caption_bbox": [100, 620, 500, 650],
      "dimensions": {
        "width": 400,
        "height": 300,
        "area": 120000
      },
      "confidence_score": 0.98,
      "reference_count": 3,
      "is_referenced": true
    }
  ],
  "metadata": {
    "total_pages": 20,
    "processing_time": 45.2,
    "total_figures": 15
  }
}
```

---

### 5. 대용량 결과 스트리밍

대용량 분석 결과를 스트리밍으로 조회합니다.

```http
GET /result/{task_id}/stream
```

**Response (200 OK):**
```
Content-Type: application/json

{
  "title": "Document Title",
  "chapters": [...],
  // 데이터가 청크 단위로 스트리밍됨
}
```

---

### 6. 특정 페이지 결과 조회

특정 페이지의 분석 결과만 조회합니다.

```http
GET /result/{task_id}/page/{page_index}
```

**Response (200 OK):**
```json
{
  "page_index": 5,
  "chapter": "2",
  "section": "2.1",
  "layouts": [
    {
      "type": "figure",
      "figure_id": "ch2_fig3",
      "bounding_box": [150, 200, 450, 500],
      "figure_caption": "Figure 3. Algorithm flowchart",
      "figure_number": 3
    }
  ],
  "full_text": "Page content..."
}
```

---

### 7. 분석 작업 취소

진행 중인 분석 작업을 취소합니다.

```http
POST /analyze/cancel/{task_id}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Analysis cancelled",
  "task_id": "uuid-string"
}
```

---

### 8. 시스템 상태 확인

서비스의 전체적인 상태를 확인합니다.

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "message": "Figure Reference System is running",
  "redis": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## 데이터 구조

### Layout Element Types

1. **Text Element**
   ```json
   {
     "type": "text",
     "bounding_box": [x_min, y_min, x_max, y_max],
     "text": "인식된 텍스트 내용",
     "confidence": 0.95
   }
   ```

2. **Figure Element**
   ```json
   {
     "type": "figure",
     "figure_id": "ch1_fig1",
     "bounding_box": [x_min, y_min, x_max, y_max],
     "figure_caption": "Figure 1. Caption text",
     "related_chapter": 1,
     "related_section": "1.2",
     "figure_number": 1
   }
   ```

3. **Figure Reference**
   ```json
   {
     "type": "figure_reference",
     "referenced_figure_id": "ch1_fig1",
     "bounding_box": [x_min, y_min, x_max, y_max],
     "reference_text": "Figure 1",
     "figure_number": 1,
     "confidence": 0.88
   }
   ```

### Bounding Box Format

모든 바운딩 박스는 `[x_min, y_min, x_max, y_max]` 형식의 좌표 배열입니다.
- `x_min, y_min`: 좌상단 좌표
- `x_max, y_max`: 우하단 좌표

---

## 에러 코드

| 코드 | 설명 | 해결 방법 |
|------|------|-----------|
| 400 | Bad Request | 요청 파라미터나 파일 형식 확인 |
| 404 | Not Found | Task ID나 리소스 경로 확인 |
| 413 | Payload Too Large | 파일 크기를 50MB 이하로 조정 |
| 500 | Internal Server Error | 서버 로그 확인 및 재시도 |

---

## 성능 최적화

### 권장 사항

1. **파일 크기**: 최적 성능을 위해 20MB 이하 권장
2. **페이지 수**: 300페이지 이하에서 최적 성능
3. **동시 요청**: 서버당 최대 2개 동시 분석 권장
4. **스트리밍 사용**: 대용량 결과는 스트리밍 API 사용

### 처리 시간

| 문서 크기 | 예상 처리 시간 |
|-----------|----------------|
| 1-10 페이지 | 30초 - 2분 |
| 11-50 페이지 | 2분 - 10분 |
| 51-100 페이지 | 10분 - 30분 |

---

## 문제 해결

### 자주 발생하는 문제

1. **분석이 시작되지 않음**
   - Redis 서버 상태 확인: `GET /health`
   - Celery worker 상태 확인

2. **진행률이 멈춤**
   - 메모리 사용량 확인 (8GB 이상 권장)
   - GPU 메모리 상태 확인

3. **Figure 참조 매핑 정확도 낮음**
   - PDF 품질 확인 (300 DPI 이상 권장)
   - 텍스트가 명확한 Figure 캡션 형식인지 확인

4. **처리 시간 과도하게 길음**
   - GPU 가속 활성화 확인
   - 문서 복잡도 및 이미지 수 확인

---