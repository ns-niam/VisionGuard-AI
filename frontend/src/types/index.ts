export type DetectionBox = { class_id: number; class_name: string; confidence: number; x1: number; y1: number; x2: number; y2: number };
export type Detection = { id: string; filename: string; image_width: number; image_height: number; detection_count: number; inference_latency_ms: number; created_at: string; detections?: DetectionBox[]; annotated_image?: string };
export type ModelInfo = { name: string; architecture: string; parameters: number; classes: { id: number; name: string }[]; validation: Record<string, number | string>; held_out_test: Record<string, number | string> };
