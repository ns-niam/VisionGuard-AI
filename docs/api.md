# API reference

The API base path is `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service and model readiness |
| GET | `/model/info` | Model provenance and verified metrics |
| POST | `/detection/image` | Multipart image inference; query `confidence`, `iou`, and `image_size` |
| GET | `/detections` | Paginated detection history; optional `search` |
| GET | `/detections/{id}` | Detection detail and annotated image data URL |
| DELETE | `/detections/{id}` | Delete one detection record |

The image endpoint accepts JPEG, PNG, WebP, and BMP up to `MAX_UPLOAD_BYTES`. Responses never expose server filesystem paths or internal exception details.
