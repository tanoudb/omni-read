import type { Project } from '../../shared/types';

interface LegacyMetadata {
  source?: string;
  dimensions?: {
    width?: number;
    height?: number;
  };
  detections?: Array<{
    class?: string;
    bbox?: [number, number, number, number];
    original?: string;
    translated?: string;
    detection_confidence?: number;
    confidence?: number;
  }>;
}

const defaultTextStyle = {
  font_family: 'Anime Ace',
  font_size: 24,
  align: 'center' as const,
  color: '#FFFFFF',
};

export const importLegacyMetadata = (legacyPayload: unknown): Project | null => {
  if (!legacyPayload || typeof legacyPayload !== 'object') {
    return null;
  }

  const payload = legacyPayload as Partial<Project> & LegacyMetadata;
  if (payload.schema_version === '1.0.0' && Array.isArray(payload.pages)) {
    return payload as Project;
  }

  if (!Array.isArray(payload.detections)) {
    return null;
  }

  const now = new Date().toISOString();
  const width = Number(payload.dimensions?.width ?? 1200);
  const height = Number(payload.dimensions?.height ?? 1600);

  const bubbles = payload.detections.map((det, idx) => {
    const [x1, y1, x2, y2] = det.bbox ?? [0, 0, 120, 80];
    return {
      id: crypto.randomUUID(),
      bbox: {
        x: Math.max(0, x1),
        y: Math.max(0, y1),
        w: Math.max(1, x2 - x1),
        h: Math.max(1, y2 - y1),
      },
      class: det.class ?? 'bulle',
      source_text: det.original ?? '',
      translated_text: det.translated ?? '',
      source_override: null,
      translated_override: null,
      llm_input_index: idx,
      llm_output_index: idx,
      detection_confidence: det.detection_confidence ?? null,
      ocr_confidence: det.confidence ?? null,
      text_style: defaultTextStyle,
      mask_strokes: [],
      errors: [],
    };
  });

  return {
    schema_version: '1.0.0',
    project_id: crypto.randomUUID(),
    name: 'Imported legacy metadata',
    created_at: now,
    updated_at: now,
    settings: {
      source_lang: 'en',
      target_lang: 'fr',
      cache_enabled: true,
    },
    pages: [
      {
        id: crypto.randomUUID(),
        index: 0,
        image_path: payload.source ?? '',
        preview_path: payload.source ?? null,
        width,
        height,
        viewport: {
          zoom: 1,
          pan_x: 0,
          pan_y: 0,
          show_translated: true,
        },
        bubbles,
      },
    ],
  };
};
