# Railway Alignment Subsystem Deployment Guide (`service/alignment`)

This subsystem adds the **Railway Alignment & Real-Time DTM** endpoints directly into your existing `service` Docker container.

---

## 1. What Has Been Added

Inside `service/`:
* **`service/alignment/`**:
  * `geospatial/kml_ingestor.py`: Ingestion of KML/KMZ corridors, UTM projection, and 20m station resampling.
  * `horizontal/pi_detector.py`: Tangent angle deflection and candidate horizontal curve detection.
  * `horizontal/curve_solver.py`: RDSO Broad Gauge transition spiral and circular curve geometry math.
  * `modules/dtm_manager.py`: Sub-pixel bilinear elevation sampling against DTM surfaces in `< 30ms`.
  * `modules/survey_reader.py`: Parsing field survey points (.csv, .xlsx, .xls) and extracting centerline polylines.
  * `job_manager.py`: In-memory async job queue with Server-Sent Events (SSE) support for `/detect`.
  * `alignment_models.py`: Pydantic request/response schemas.
  * `alignment_router.py`: FastAPI router exposing all endpoints.
  * `test_alignment_service.py`: Automated unit test suite.
* **`service/requirements.txt`**: Added `shapely`, `pyproj`, `pandas`, `openpyxl`, `scipy`, `rasterio`, and `python-multipart`.
* **`service/main.py`**: Added `service/alignment` to `sys.path` and registered `alignment_router`.

---

## 2. Deploying on Your Ubuntu Server

On your Ubuntu development server, pull or upload the updated `service/` directory and rebuild the container:

```bash
# 1. Navigate to the service folder on your Ubuntu server
cd /path/to/service

# 2. Rebuild and restart the container
docker compose down
docker compose up -d --build

# 3. Verify logs
docker compose logs -f
```

---

## 3. Registered Endpoints & Testing

All endpoints are hosted under your existing server root path (`https://bridge-gad-dev.aagento.ai/api/service`):

### A. Horizontal Alignment Detection
* **URL**: `POST /api/v1/alignment-studio/detect`
* **Test Command**:
  ```bash
  curl -X POST "https://bridge-gad-dev.aagento.ai/api/service/api/v1/alignment-studio/detect" \
       -F "file=@alignment.kml" \
       -F "simplification_tolerance_m=5.0" \
       -F "minimum_deflection_deg=1.0"
  ```
* **Returns**: `{"job_id": "job_xxxxxxxxxx"}`

### B. Job Status & Results
* **URL**: `GET /api/v1/jobs/{job_id}/status`
* **Test Command**:
  ```bash
  curl "https://bridge-gad-dev.aagento.ai/api/service/api/v1/jobs/{job_id}/status"
  ```
* **Returns**: Alignment stations, candidate PIs, raw lines, and detected curve geometry.

### C. Real-Time DTM Elevation Query (< 30ms)
* **URL**: `POST /api/v1/alignment-studio/dtm/query`
* **Test Command**:
  ```bash
  curl -X POST "https://bridge-gad-dev.aagento.ai/api/service/api/v1/alignment-studio/dtm/query" \
       -H "Content-Type: application/json" \
       -d '{"project_id": "test_project", "coordinates": [[87.75, 21.78], [87.76, 21.79]]}'
  ```

### D. Survey Centerline Extraction
* **URL**: `POST /api/v1/alignment-studio/survey/generate-centerline`
* **Test Command**:
  ```bash
  curl -X POST "https://bridge-gad-dev.aagento.ai/api/service/api/v1/alignment-studio/survey/generate-centerline" \
       -F "file=@survey.csv" \
       -F "chainage_interval_m=20.0"
  ```

