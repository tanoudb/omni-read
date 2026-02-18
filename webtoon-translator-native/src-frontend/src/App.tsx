import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import ImageListPanel from './features/image-list/ImageListPanel';
import CanvasEditor from './features/canvas-editor/CanvasEditor';
import BubbleInspectorPanel from './features/bubble-inspector/BubbleInspectorPanel';
import SettingsPanel from './features/settings/SettingsPanel';
import { useCanvasStore } from './features/canvas-editor/canvasStore';
import { useImageListStore } from './features/image-list/imageListStore';
import { useProjectStore } from './features/project/projectStore';
import { deserializeProject } from './features/project/projectSerializer';
import { importLegacyMetadata } from './features/project/projectImporter';
import type { Page, Project } from './shared/types';
import { useJobsStore } from './features/jobs/jobsStore';
import { createJob, pollJob, runDetect, runOcr, runTranslate } from './features/jobs/manualApiService';
import { renderPagePreview } from './features/render/renderService';
import type { Bubble } from './shared/types';
import { invoke } from '@tauri-apps/api/core';

const makeEmptyPageFromFile = async (file: File, index: number, copiedPath: string): Promise<Page> => {
  const previewSrc = URL.createObjectURL(file);
  const dimensions = await new Promise<{ width: number; height: number }>((resolve) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      resolve({ width: 1200, height: 1600 });
    };
    img.src = previewSrc;
  });

  return {
    id: crypto.randomUUID(),
    index,
    image_path: copiedPath,
    preview_path: previewSrc,
    width: dimensions.width,
    height: dimensions.height,
    viewport: {
      zoom: 1,
      pan_x: 0,
      pan_y: 0,
      show_translated: true,
    },
    bubbles: [],
  };
};

const normalizeProjectForUi = (project: Project): Project => project;

const FIXED_OUTPUT_DIR = 'A:/omni read/temp/output';

const isAbsoluteSystemPath = (value: string): boolean => /^[A-Za-z]:\//.test(value.replace(/\\/g, '/'));

const toNumber = (value: unknown, fallback = 0): number => {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
};

const parseBbox = (detection: Record<string, unknown>): { x: number; y: number; w: number; h: number } => {
  const rawBbox = detection.bbox;

  if (Array.isArray(rawBbox) && rawBbox.length >= 4) {
    const a = toNumber(rawBbox[0], 0);
    const b = toNumber(rawBbox[1], 0);
    const c = toNumber(rawBbox[2], a + 120);
    const d = toNumber(rawBbox[3], b + 80);

    const isXyxy = c > a && d > b;
    if (isXyxy) {
      return {
        x: Math.max(0, a),
        y: Math.max(0, b),
        w: Math.max(1, c - a),
        h: Math.max(1, d - b),
      };
    }

    return {
      x: Math.max(0, a),
      y: Math.max(0, b),
      w: Math.max(1, c),
      h: Math.max(1, d),
    };
  }

  if (typeof rawBbox === 'string') {
    const parts = rawBbox
      .split(/[\s,;]+/)
      .map((part) => Number(part))
      .filter((value) => Number.isFinite(value));
    if (parts.length >= 4) {
      const [a, b, c, d] = parts;
      const isXyxy = c > a && d > b;
      return {
        x: Math.max(0, a),
        y: Math.max(0, b),
        w: Math.max(1, isXyxy ? c - a : c),
        h: Math.max(1, isXyxy ? d - b : d),
      };
    }
  }

  if (rawBbox && typeof rawBbox === 'object') {
    const bboxObject = rawBbox as Record<string, unknown>;
    const x = toNumber(bboxObject.x ?? bboxObject.left ?? bboxObject.x1, 0);
    const y = toNumber(bboxObject.y ?? bboxObject.top ?? bboxObject.y1, 0);
    const widthCandidate = bboxObject.w ?? bboxObject.width;
    const heightCandidate = bboxObject.h ?? bboxObject.height;
    const x2 = toNumber(bboxObject.x2, x + 120);
    const y2 = toNumber(bboxObject.y2, y + 80);

    const width =
      widthCandidate !== undefined
        ? toNumber(widthCandidate, 120)
        : Math.max(1, x2 - x);
    const height =
      heightCandidate !== undefined
        ? toNumber(heightCandidate, 80)
        : Math.max(1, y2 - y);

    return {
      x: Math.max(0, x),
      y: Math.max(0, y),
      w: Math.max(1, width),
      h: Math.max(1, height),
    };
  }

  return { x: 0, y: 0, w: 120, h: 80 };
};

const mapDetectionsToBubbles = (detections: Array<Record<string, unknown>>): Bubble[] => {
  return detections.map((det, index) => {
    const parsed = parseBbox(det);

    return {
      id: crypto.randomUUID(),
      bbox: {
        x: parsed.x,
        y: parsed.y,
        w: parsed.w,
        h: parsed.h,
      },
      class: String(det.class ?? 'bulle'),
      source_text: String(det.original ?? ''),
      translated_text: String(det.translated ?? ''),
      source_override: null,
      translated_override: null,
      llm_input_index: index,
      llm_output_index: index,
      detection_confidence:
        det.detection_confidence !== undefined ? Number(det.detection_confidence) : null,
      ocr_confidence: det.confidence !== undefined ? Number(det.confidence) : null,
      text_style: {
        font_family: 'Anime Ace',
        font_size: 24,
        align: 'center',
        color: '#FFFFFF',
      },
      mask_strokes: [],
      errors: [],
    };
  });
};

const extractDetections = (result: unknown): Array<Record<string, unknown>> => {
  if (!result || typeof result !== 'object') {
    return [];
  }

  const payload = result as Record<string, unknown>;
  const metadata = payload.metadata as Record<string, unknown> | undefined;
  const nestedResult = payload.result as Record<string, unknown> | undefined;
  const data = payload.data as Record<string, unknown> | undefined;

  const candidates: unknown[] = [
    payload.detections,
    payload.bubbles,
    metadata?.detections,
    metadata?.bubbles,
    nestedResult,
    nestedResult?.detections,
    nestedResult?.bubbles,
    data?.detections,
    data?.bubbles,
  ];

  const asDetectionsArray = (candidate: unknown): Array<Record<string, unknown>> | null => {
    if (!Array.isArray(candidate)) {
      return null;
    }
    return candidate.filter((item): item is Record<string, unknown> => !!item && typeof item === 'object');
  };

  for (const candidate of candidates) {
    const parsed = asDetectionsArray(candidate);
    if (parsed && parsed.length > 0) {
      return parsed;
    }
  }

  for (const candidate of candidates) {
    const parsed = asDetectionsArray(candidate);
    if (parsed) {
      return parsed;
    }
  }

  return [];
};

const App = () => {
  const [mode, setMode] = useState<'auto' | 'manuel'>('auto');
  const [languagePair, setLanguagePair] = useState('EN→FR');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isFileMenuOpen, setIsFileMenuOpen] = useState(false);
  const [toasts, setToasts] = useState<Array<{ id: string; type: 'error' | 'success'; message: string }>>([]);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const projectInputRef = useRef<HTMLInputElement | null>(null);

  const project = useProjectStore((state) => state.project);
  const loadProject = useProjectStore((state) => state.loadProject);
  const setPageBubbles = useProjectStore((state) => state.setPageBubbles);
  const patchPageBubbles = useProjectStore((state) => state.patchPageBubbles);
  const setPagePreviewPath = useProjectStore((state) => state.setPagePreviewPath);

  const setPages = useImageListStore((state) => state.setPages);
  const selectPage = useImageListStore((state) => state.selectPage);

  const activePageId = useCanvasStore((state) => state.activePageId);
  const activeBubbleId = useCanvasStore((state) => state.activeBubbleId);
  const zoomEnabled = useCanvasStore((state) => state.zoomEnabled);
  const setActivePage = useCanvasStore((state) => state.setActivePage);
  const setActiveBubble = useCanvasStore((state) => state.setActiveBubble);
  const setZoomEnabled = useCanvasStore((state) => state.setZoomEnabled);

  const jobStatus = useJobsStore((state) => state.status);
  const jobLogs = useJobsStore((state) => state.logs);
  const startJob = useJobsStore((state) => state.startJob);
  const pushLogs = useJobsStore((state) => state.pushLogs);
  const setJobStatus = useJobsStore((state) => state.setStatus);
  const setJobError = useJobsStore((state) => state.setError);
  const resetJob = useJobsStore((state) => state.reset);

  const activePage = project?.pages.find((page) => page.id === activePageId) ?? null;

  const showToast = (type: 'error' | 'success', message: string) => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { id, type, message }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 3600);
  };

  useEffect(() => {
    const pages = project?.pages ?? [];
    setPages(pages);

    if (!pages.length) {
      setActivePage(null);
      selectPage(null);
      setActiveBubble(null);
      return;
    }

    const nextActiveId = activePageId && pages.some((page) => page.id === activePageId)
      ? activePageId
      : pages[0].id;

    setActivePage(nextActiveId);
    selectPage(nextActiveId);
    setActiveBubble(null);
  }, [project, activePageId, selectPage, setActiveBubble, setActivePage, setPages]);

  const handleNewProjectClick = () => {
    setIsFileMenuOpen(false);
    imageInputRef.current?.click();
  };

  const handleImagesSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) {
      return;
    }

    const copiedPaths: string[] = [];
    for (const file of files) {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const savedPath = await invoke<string>('save_uploaded_image', {
          filename: file.name,
          data: Array.from(uint8Array),
        });
        copiedPaths.push(savedPath);
      } catch (error) {
        showToast('error', error instanceof Error ? error.message : `Copie échouée pour ${file.name}`);
        event.target.value = '';
        return;
      }
    }

    const pages = await Promise.all(
      files.map((file, index) => makeEmptyPageFromFile(file, index, copiedPaths[index]))
    );
    const now = new Date().toISOString();

    const nextProject: Project = {
      schema_version: '1.0.0',
      project_id: crypto.randomUUID(),
      name: `Project ${now.slice(0, 10)}`,
      created_at: now,
      updated_at: now,
      settings: {
        source_lang: 'en',
        target_lang: 'fr',
        cache_enabled: true,
      },
      pages,
    };

    loadProject(nextProject);
    event.target.value = '';
  };

  const handleOpenProjectClick = () => {
    setIsFileMenuOpen(false);
    projectInputRef.current?.click();
  };

  const handleProjectSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const raw = await file.text();

    let nextProject: Project | null = null;
    try {
      nextProject = deserializeProject(raw);
    } catch {
      try {
        const parsed = JSON.parse(raw);
        nextProject = importLegacyMetadata(parsed);
      } catch {
        nextProject = null;
      }
    }

    if (nextProject) {
      loadProject(normalizeProjectForUi(nextProject));
    } else {
      showToast('error', 'Impossible de parser le project.json');
    }

    event.target.value = '';
  };

  const handleAutoRun = async () => {
    if (!project || project.pages.length === 0) {
      showToast('error', 'Aucun projet/page disponible pour le mode Auto.');
      return;
    }

    try {
      resetJob();
      for (const page of project.pages) {
        if (!page.image_path || !isAbsoluteSystemPath(page.image_path)) {
          showToast('error', `Path système manquant pour Auto: page ${page.index + 1}`);
          continue;
        }

        const create = await createJob({
          input_path: page.image_path,
          output_dir: FIXED_OUTPUT_DIR,
          debug: false,
        });

        startJob(create.job_id);
        setJobStatus('running');

        const finalState = await pollJob(create.job_id, (snapshot) => {
          if (snapshot.logs?.length) {
            pushLogs(
              snapshot.logs.map((log) => ({
                ...log,
                message: `[Page ${page.index + 1}] ${log.message}`,
              }))
            );
          }
          setJobStatus(snapshot.status);
        }, 500);

        if (finalState.status === 'failed') {
          throw new Error(finalState.error ?? `Auto job failed (page ${page.index + 1})`);
        }

        console.log('[AutoRun] job result', {
          pageId: page.id,
          pageIndex: page.index,
          result: finalState.result,
          detectionsRaw: finalState.result?.detections,
          detectionsType: typeof finalState.result?.detections,
          detectionsIsArray: Array.isArray(finalState.result?.detections),
        });
        console.log(
          '[AutoRun] Full result keys:',
          finalState.result && typeof finalState.result === 'object'
            ? Object.keys(finalState.result as Record<string, unknown>)
            : []
        );
        console.log('[AutoRun] Full result:', JSON.stringify(finalState.result, null, 2));

        const detections = extractDetections(finalState.result);
        const mappedBubbles = mapDetectionsToBubbles(detections);

        console.log('[AutoRun] mapped bubbles', {
          pageId: page.id,
          count: mappedBubbles.length,
          firstThree: mappedBubbles.slice(0, 3).map((bubble) => ({
            id: bubble.id,
            bbox: bubble.bbox,
            source: bubble.source_text,
            translated: bubble.translated_text,
          })),
        });

        console.log('[AutoRun] calling setPageBubbles', {
          pageId: page.id,
          bubblesCount: mappedBubbles.length,
        });
        setPageBubbles(page.id, mappedBubbles);

        if (typeof finalState.result?.output === 'string') {
          setPagePreviewPath(page.id, finalState.result.output);
        }
      }

      showToast('success', 'Mode Auto terminé. Bubbles mises à jour.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Erreur mode Auto';
      setJobError(message);
      setJobStatus('failed');
      showToast('error', message);
    }
  };

  const handleDetect = async () => {
    if (!activePage) {
      showToast('error', 'Sélectionne une page active avant Detect.');
      return;
    }

    try {
      const response = await runDetect({
        image_path: activePage.image_path,
        classes: ['bulle', 'out_text', 'System'],
        debug: false,
      });
      setPageBubbles(activePage.id, response.bubbles ?? []);
      if (response.errors?.length) {
        showToast('error', response.errors.map((e) => e.message).join(' | '));
      } else {
        showToast('success', 'Detect terminé.');
      }
    } catch (error) {
      showToast('error', error instanceof Error ? error.message : 'Detect failed');
    }
  };

  const handleOcr = async () => {
    if (!activePage) {
      showToast('error', 'Sélectionne une page active avant OCR.');
      return;
    }

    try {
      const response = await runOcr({
        image_path: activePage.image_path,
        bubbles: activePage.bubbles,
      });

      patchPageBubbles(
        activePage.id,
        (response.bubbles ?? []).map((bubble) => ({
          id: bubble.id,
          source_text: bubble.source_text,
          ocr_confidence: bubble.ocr_confidence,
          errors: bubble.errors,
        }))
      );

      if (response.errors?.length) {
        showToast('error', response.errors.map((e) => e.message).join(' | '));
      } else {
        showToast('success', 'OCR terminé.');
      }
    } catch (error) {
      showToast('error', error instanceof Error ? error.message : 'OCR failed');
    }
  };

  const handleTranslate = async () => {
    if (!activePage) {
      showToast('error', 'Sélectionne une page active avant Translate.');
      return;
    }

    try {
      const response = await runTranslate({
        bubbles: activePage.bubbles.map((bubble) => ({
          id: bubble.id,
          source_text: bubble.source_override ?? bubble.source_text,
          translated_text: bubble.translated_override ?? bubble.translated_text,
          llm_input_index: bubble.llm_input_index,
          llm_output_index: bubble.llm_output_index,
        })),
        cache_enabled: true,
        return_llm_debug: true,
      });

      patchPageBubbles(
        activePage.id,
        (response.bubbles ?? []).map((bubble) => ({
          id: bubble.id,
          translated_text: bubble.translated_text,
          llm_input_index: bubble.llm_input_index,
          llm_output_index: bubble.llm_output_index,
          errors: bubble.errors,
        }))
      );

      if (response.errors?.length) {
        showToast('error', response.errors.map((e) => e.message).join(' | '));
      } else {
        showToast('success', 'Translate terminé.');
      }
    } catch (error) {
      showToast('error', error instanceof Error ? error.message : 'Translate failed');
    }
  };

  const handleRender = async () => {
    if (!activePage) {
      showToast('error', 'Sélectionne une page active avant Render.');
      return;
    }

    try {
      const response = await renderPagePreview({
        image_path: activePage.image_path,
        bubbles: activePage.bubbles.map((bubble) => ({
          ...bubble,
          source_text: bubble.source_override ?? bubble.source_text,
          translated_text: bubble.translated_override ?? bubble.translated_text,
        })),
        text_only: false,
        skip_inpainting: false,
      });
      setPagePreviewPath(activePage.id, response.preview_path ?? null);

      if (response.errors?.length) {
        showToast('error', response.errors.map((e) => e.message).join(' | '));
      } else {
        showToast('success', 'Render terminé.');
      }
    } catch (error) {
      showToast('error', error instanceof Error ? error.message : 'Render failed');
    }
  };

  return (
    <div className="app-shell">
      <input
        ref={imageInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        multiple
        hidden
        onChange={handleImagesSelected}
      />

      <input
        ref={projectInputRef}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={handleProjectSelected}
      />

      <header className="topbar">
        <div className="topbar-group topbar-menus">
          <div className="menu-wrapper">
            <button
              className="menu-btn"
              type="button"
              onClick={() => setIsFileMenuOpen((value) => !value)}
            >
              File
            </button>
            {isFileMenuOpen ? (
              <div className="menu-dropdown">
                <button className="menu-item" type="button" onClick={handleNewProjectClick}>
                  New Project
                </button>
                <button className="menu-item" type="button" onClick={handleOpenProjectClick}>
                  Open Project
                </button>
                <button className="menu-item" type="button">Save</button>
                <button className="menu-item" type="button">Export</button>
              </div>
            ) : null}
          </div>
          <button className="menu-btn" type="button">Edit</button>
          <button className="menu-btn" type="button">View</button>
        </div>

        <div className="topbar-group topbar-controls">
          <label className="control-label" htmlFor="mode-select">
            Mode
          </label>
          <select
            id="mode-select"
            className="control-select"
            value={mode}
            onChange={(event) => setMode(event.target.value as 'auto' | 'manuel')}
          >
            <option value="auto">Auto</option>
            <option value="manuel">Manuel</option>
          </select>

          <label className="control-label" htmlFor="lang-select">
            Langue
          </label>
          <select
            id="lang-select"
            className="control-select"
            value={languagePair}
            onChange={(event) => setLanguagePair(event.target.value)}
          >
            <option value="EN→FR">EN→FR</option>
          </select>
        </div>

        <div className="topbar-group topbar-actions">
          <button className="action-btn" type="button" onClick={handleAutoRun}>
            Auto Run
          </button>
          <button
            className="action-btn"
            type="button"
            onClick={() => setZoomEnabled(!zoomEnabled)}
          >
            {zoomEnabled ? 'Disable Zoom' : 'Enable Zoom'}
          </button>
          <button className="action-btn" type="button" onClick={handleDetect}>Detect</button>
          <button className="action-btn" type="button" onClick={handleOcr}>OCR</button>
          <button className="action-btn" type="button" onClick={handleTranslate}>Translate</button>
          <button className="action-btn" type="button" onClick={handleRender}>Render</button>
          <button
            className="icon-btn"
            type="button"
            onClick={() => setIsSettingsOpen(true)}
            aria-label="Open settings"
          >
            ⚙️
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className="left-sidebar">
          <ImageListPanel />
        </aside>

        <section className="canvas-area">
          <CanvasEditor />

          <aside className="logs-panel">
            <h4>Logs ({jobStatus})</h4>
            <div className="logs-scroll">
              {jobLogs.length === 0 ? <p>Aucun log.</p> : null}
              {jobLogs.map((log, idx) => (
                <p key={`${log.ts}-${idx}`}>
                  [{log.level}] {log.message}
                </p>
              ))}
            </div>
          </aside>

          {activeBubbleId ? (
            <aside className="properties-bottom">
              <BubbleInspectorPanel />
            </aside>
          ) : null}
        </section>
      </main>

      <div className="toast-stack">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.type}`}>
            {toast.message}
          </div>
        ))}
      </div>

      {isSettingsOpen ? (
        <SettingsPanel onClose={() => setIsSettingsOpen(false)} />
      ) : null}
    </div>
  );
};

export default App;
