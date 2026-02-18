import type { Bubble, LlmDebug } from './domain';

export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
  bubble_id?: string;
}

export interface AutoJobCreateRequest {
  input_path: string;
  output_dir: string;
  debug: boolean;
}

export interface AutoJobCreateResponse {
  job_id: string;
  status: 'queued' | 'running' | 'done' | 'failed';
}

export interface JobLogLine {
  ts: string;
  level: string;
  message: string;
}

export interface AutoJobStatusResponse {
  job_id: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  logs: JobLogLine[];
  next_offset: number;
  result: {
    output?: string;
    detections?: Array<{
      class?: string;
      bbox?: [number, number, number, number] | number[];
      original?: string;
      translated?: string;
      detection_confidence?: number;
      confidence?: number;
      source_lang_detected?: string;
      source_lang_confidence?: number;
      global_confidence?: number;
    }>;
  } | null;
  error: string | null;
}

export interface DetectRequest {
  image_path: string;
  classes: string[];
  debug: boolean;
}

export interface DetectResponse {
  page: {
    width: number;
    height: number;
  };
  bubbles: Bubble[];
  errors: ApiError[];
}

export interface OcrRequest {
  image_path: string;
  bubbles: Bubble[];
}

export interface OcrResponse {
  bubbles: Array<Pick<Bubble, 'id' | 'source_text' | 'ocr_confidence' | 'errors'>>;
  errors: ApiError[];
}

export interface TranslateRequest {
  bubbles: Array<Pick<Bubble, 'id' | 'source_text' | 'translated_text' | 'llm_input_index' | 'llm_output_index'>>;
  cache_enabled: boolean;
  return_llm_debug: boolean;
}

export interface TranslateResponse {
  bubbles: Array<Pick<Bubble, 'id' | 'translated_text' | 'llm_input_index' | 'llm_output_index' | 'errors'>>;
  llm_debug: LlmDebug | null;
  errors: ApiError[];
}

export interface RemapRequest {
  page_id: string;
  remap: Array<{ bubble_id: string; output_index: number }>;
}

export interface RemapResponse {
  bubbles: Array<Pick<Bubble, 'id' | 'translated_text' | 'llm_output_index' | 'errors'>>;
  errors: ApiError[];
}

export interface RenderRequest {
  image_path: string;
  bubbles: Bubble[];
  text_only: boolean;
  skip_inpainting: boolean;
}

export interface RenderResponse {
  preview_path: string;
  timings: {
    text_render_ms: number;
    inpaint_ms: number;
    total_ms: number;
  };
  errors: ApiError[];
}

export interface CacheToggleRequest {
  enabled: boolean;
}

export interface CacheToggleResponse {
  enabled: boolean;
}
