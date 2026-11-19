# Railway Alignment Detection Subsystem Deployment Guide (`alignment`)

This container runs the **Railway Horizontal Alignment Detection Microservice** adhering to RDSO / Indian Railways BG standards.

---

## 1. Exposed Endpoints

Only the **`/detect`** endpoint is exposed:
* **Primary URL**: `POST /detect`
* **Alias URL**: `POST /api/v1/alignment-studio/detect`
* **Input**: Multipart Form Data
  * `file`: `.kml` or `.kmz` railway corridor polyline file (*required*)
  * `simplification_tolerance_m`: Noise cleanup tolerance in meters (default: `5.0`)
  * `minimum_deflection_deg`: Minimum bend turn angle in degrees (default: `1.0`)
  * `start_chainage_m`: Starting chainage in meters (default: `0.0`)
  * `line_index`: Selected corridor index if multi-line KML (default: `0`)
* **Output (HTTP 200)**: Complete synchronous alignment solution:
  * Projected UTM EPSG zone
  * 20m pegged station chainages (`KM+MMM.MM`)
  * Candidate Points of Intersection (PI) with deflection angles & curve radii
  * Tangent lines & bearings
  * Full RDSO transition spiral & circular arc geometry

*All other endpoints (`/dtm/query`, `/survey/...`, `/jobs/...`) are disabled/unexposed.*

---

## 2. Managing the Container

To build and run the container on port `8001`:

```bash
cd alignment
docker compose up -d --build
```

To view logs:
```bash
docker logs -f railway_alignment_service
```

---

## 3. Testing the Endpoint

```bash
curl -X POST "http://127.0.0.1:8001/detect" \
     -F "file=@alignment.kml" \
     -F "simplification_tolerance_m=5.0" \
     -F "minimum_deflection_deg=1.0"
```

