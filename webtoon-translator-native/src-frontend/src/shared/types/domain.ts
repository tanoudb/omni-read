export type UUID = string;

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface TextStyle {
  font_family: string;
  font_size: number;
  align: 'left' | 'center' | 'right';
  color: string;
}

export interface MaskPoint {
  x: number;
  y: number;
}

export interface MaskStroke {
  id: UUID;
  size: number;
  points: MaskPoint[];
}

export interface BubbleError {
  code: string;
  message: string;
}

export interface Bubble {
  id: UUID;
  bbox: BBox;
  class: string;
  source_text: string;
  translated_text: string;
  source_override: string | null;
  translated_override: string | null;
  llm_input_index: number | null;
  llm_output_index: number | null;
  detection_confidence: number | null;
  ocr_confidence: number | null;
  text_style: TextStyle;
  mask_strokes: MaskStroke[];
  errors: BubbleError[];
}

export interface ViewportState {
  zoom: number;
  pan_x: number;
  pan_y: number;
  show_translated: boolean;
}

export interface LlmMappingRow {
  input_index: number;
  output_index: number;
  bubble_id: UUID;
}

export interface LlmDebug {
  payload: Record<string, unknown> | null;
  raw_response: string | null;
  parsed_mapping: LlmMappingRow[];
}

export interface Page {
  id: UUID;
  index: number;
  image_path: string;
  preview_path: string | null;
  width: number;
  height: number;
  viewport: ViewportState;
  bubbles: Bubble[];
  llm_debug?: LlmDebug;
}

export interface ProjectSettings {
  source_lang: string;
  target_lang: string;
  cache_enabled: boolean;
  render?: {
    skip_inpainting_on_text_only?: boolean;
  };
}

export interface Project {
  schema_version: '1.0.0';
  project_id: UUID;
  name: string;
  created_at: string;
  updated_at: string;
  settings: ProjectSettings;
  pages: Page[];
}
