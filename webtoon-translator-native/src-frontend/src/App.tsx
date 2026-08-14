import { useEffect, useRef, useState, useCallback, type ChangeEvent } from 'react';
import ImageListPanel from './features/image-list/ImageListPanel';
import CanvasEditor from './features/canvas-editor/CanvasEditor';
import BubbleInspectorPanel from './features/bubble-inspector/BubbleInspectorPanel';
import SettingsPanel from './features/settings/SettingsPanel';
import BatchManager from './features/batch-processing/BatchManager';
import GlossaryManager from './features/glossary/GlossaryManager';
import ExportModal from './features/export/ExportModal';
import { useCanvasStore } from './features/canvas-editor/canvasStore';
import { useImageListStore } from './features/image-list/imageListStore';
import { useProjectStore } from './features/project/projectStore';
import { deserializeProject } from './features/project/projectSerializer';
import { importLegacyMetadata } from './features/project/projectImporter';
import { useAutosave } from './features/project/projectAutosave';
import type { Page, Project, Bubble } from './shared/types';
import { useJobsStore } from './features/jobs/jobsStore';
import { createJob, pollJob, runDetect, runOcr, runTranslate } from './features/jobs/manualApiService';
import { renderPagePreview } from './features/render/renderService';
import { invoke } from '@tauri-apps/api/core';
import { useHistoryStore } from './shared/history/historyStore';
import { enablePatches } from 'immer';

enablePatches();

// ─── Helpers ────────────────────────────────────────────────────────────────

const FIXED_OUTPUT_DIR = 'A:/omni read/temp/output';

const isAbsoluteSystemPath = (v: string) => /^[A-Za-z]:\//.test(v.replace(/\\/g, '/'));
const toNumber = (v: unknown, fb = 0) => { const n = Number(v); return Number.isFinite(n) ? n : fb; };

const parseBbox = (det: Record<string, unknown>) => {
  const raw = det.bbox;
  if (Array.isArray(raw) && raw.length >= 4) {
    const [a, b, c, d] = raw.map((x) => toNumber(x));
    return c > a && d > b
      ? { x: Math.max(0, a), y: Math.max(0, b), w: Math.max(1, c - a), h: Math.max(1, d - b) }
      : { x: Math.max(0, a), y: Math.max(0, b), w: Math.max(1, c), h: Math.max(1, d) };
  }
  if (raw && typeof raw === 'object') {
    const o = raw as Record<string, unknown>;
    const x = toNumber(o.x ?? o.x1, 0), y = toNumber(o.y ?? o.y1, 0);
    const w = o.w != null ? toNumber(o.w) : Math.max(1, toNumber(o.x2, x + 120) - x);
    const h = o.h != null ? toNumber(o.h) : Math.max(1, toNumber(o.y2, y + 80) - y);
    return { x: Math.max(0, x), y: Math.max(0, y), w: Math.max(1, w), h: Math.max(1, h) };
  }
  return { x: 0, y: 0, w: 120, h: 80 };
};

const mapDetectionsToBubbles = (dets: Array<Record<string, unknown>>): Bubble[] =>
  dets.map((det, i) => ({
    id: crypto.randomUUID(),
    bbox: parseBbox(det),
    class: String(det.class ?? 'bulle'),
    source_text: String(det.original ?? ''),
    translated_text: String(det.translated ?? ''),
    source_override: null,
    translated_override: null,
    llm_input_index: i,
    llm_output_index: i,
    detection_confidence: det.detection_confidence != null ? Number(det.detection_confidence) : null,
    ocr_confidence: det.confidence != null ? Number(det.confidence) : null,
    text_style: { font_family: 'Anime Ace', font_size: 24, align: 'center', color: '#000000' },
    mask_strokes: [],
    errors: [],
  }));

const extractDetections = (result: unknown): Array<Record<string, unknown>> => {
  if (!result || typeof result !== 'object') return [];
  const p = result as Record<string, unknown>;
  const candidates = [p.detections, p.bubbles, (p.metadata as any)?.detections, (p.metadata as any)?.bubbles, (p.result as any)?.bubbles, (p.result as any)?.detections];
  for (const c of candidates) {
    if (Array.isArray(c) && c.length > 0) return c.filter(x => !!x && typeof x === 'object');
  }
  return [];
};

const makeEmptyPage = async (file: File, index: number, copiedPath: string): Promise<Page> => {
  const src = URL.createObjectURL(file);
  const dims = await new Promise<{ width: number; height: number }>((res) => {
    const img = new Image();
    img.onload = () => res({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => res({ width: 1200, height: 1600 });
    img.src = src;
  });
  return {
    id: crypto.randomUUID(), index,
    image_path: copiedPath, preview_path: src,
    width: dims.width, height: dims.height,
    viewport: { zoom: 1, pan_x: 0, pan_y: 0, show_translated: true },
    bubbles: [],
  };
};

// ─── Tool definitions ────────────────────────────────────────────────────────

const TOOLS = [
  { id: 'select', icon: '↖', title: 'Sélect (V)' },
  { id: 'draw',   icon: '⬜', title: 'Dessiner bbox (B)' },
  { id: 'pan',    icon: '✋', title: 'Pan (Espace)' },
  { id: 'delete', icon: '✕', title: 'Supprimer (Del)' },
] as const;

type PipelineStepId = 'detect' | 'ocr' | 'translate' | 'render';
type StepStatus = 'idle' | 'running' | 'done' | 'error';

const PIPELINE_STEPS: { id: PipelineStepId; label: string }[] = [
  { id: 'detect', label: 'Detect' },
  { id: 'ocr', label: 'OCR' },
  { id: 'translate', label: 'Translate' },
  { id: 'render', label: 'Render' },
];

// ─── App ─────────────────────────────────────────────────────────────────────

const App = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isBatchManagerOpen, setIsBatchManagerOpen] = useState(false);
  const [isGlossaryOpen, setIsGlossaryOpen] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [isFileMenuOpen, setIsFileMenuOpen] = useState(false);
  const [toasts, setToasts] = useState<Array<{ id: string; type: 'error' | 'success'; message: string }>>([]);
  const [stepStatus, setStepStatus] = useState<Record<PipelineStepId, StepStatus>>({
    detect: 'idle', ocr: 'idle', translate: 'idle', render: 'idle',
  });

  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const projectInputRef = useRef<HTMLInputElement | null>(null);

  // Stores
  const project = useProjectStore((s) => s.project);
  const loadProject = useProjectStore((s) => s.loadProject);
  const setPageBubbles = useProjectStore((s) => s.setPageBubbles);
  const patchPageBubbles = useProjectStore((s) => s.patchPageBubbles);
  const setPagePreviewPath = useProjectStore((s) => s.setPagePreviewPath);

  const setPages = useImageListStore((s) => s.setPages);
  const selectPage = useImageListStore((s) => s.selectPage);

  const activePageId = useCanvasStore((s) => s.activePageId);
  const activeBubbleId = useCanvasStore((s) => s.activeBubbleId);
  const tool = useCanvasStore((s) => s.tool);
  const showOriginal = useCanvasStore((s) => s.showOriginal);
  const setActivePage = useCanvasStore((s) => s.setActivePage);
  const setActiveBubble = useCanvasStore((s) => s.setActiveBubble);
  const setTool = useCanvasStore((s) => s.setTool);
  const toggleShowOriginal = useCanvasStore((s) => s.toggleShowOriginal);

  const jobStatus = useJobsStore((s) => s.status);
  const jobLogs = useJobsStore((s) => s.logs);
  const startJob = useJobsStore((s) => s.startJob);
  const pushLogs = useJobsStore((s) => s.pushLogs);
  const setJobStatus = useJobsStore((s) => s.setStatus);
  const setJobError = useJobsStore((s) => s.setError);
  const resetJob = useJobsStore((s) => s.reset);

  const activePage = project?.pages.find((p) => p.id === activePageId) ?? null;

  useAutosave();

  // Toast helper
  const showToast = useCallback((type: 'error' | 'success', message: string) => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
  }, []);

  // Sync pages
  useEffect(() => {
    const pages = project?.pages ?? [];
    setPages(pages);
    if (!pages.length) { 
      if (activePageId !== null) {
        setActivePage(null); 
        selectPage(null); 
        setActiveBubble(null); 
      }
      return; 
    }
    const nextId = activePageId && pages.some((p) => p.id === activePageId) ? activePageId : pages[0].id;
    if (nextId !== activePageId) {
      setActivePage(nextId);
      selectPage(nextId);
    }
  }, [project, activePageId, selectPage, setActiveBubble, setActivePage, setPages]);

  // Reset pipeline steps when switching pages
  useEffect(() => {
    setStepStatus({ detect: 'idle', ocr: 'idle', translate: 'idle', render: 'idle' });
  }, [activePageId]);

  const undo = useHistoryStore((s) => s.undo);
  const redo = useHistoryStore((s) => s.redo);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'TEXTAREA' || document.activeElement?.tagName === 'INPUT') return;
      if (e.key === 'v' || e.key === 'V') setTool('select');
      if (e.key === 'b' || e.key === 'B') setTool('draw');
      if (e.key === 'h' || e.key === 'H' || e.key === ' ') {
        if (e.key === ' ' && e.type === 'keydown') e.preventDefault();
        setTool('pan');
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (activePageId && activeBubbleId && project) {
          const page = project.pages.find(p => p.id === activePageId);
          if (page) {
            setPageBubbles(activePageId, page.bubbles.filter(b => b.id !== activeBubbleId));
            setActiveBubble(null);
          }
        }
      }
      if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        undo();
      }
      if (e.ctrlKey && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault();
        redo();
      }
      if (e.ctrlKey && (e.key === 's' || e.key === 'S')) { 
        e.preventDefault(); 
        useProjectStore.getState().saveProject(); 
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setTool, activePageId, activeBubbleId, project, setPageBubbles, setActiveBubble, undo, redo]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!isFileMenuOpen) return;
    const close = () => setIsFileMenuOpen(false);
    setTimeout(() => window.addEventListener('click', close), 0);
    return () => window.removeEventListener('click', close);
  }, [isFileMenuOpen]);

  // ── File handlers ────────────────────────────────────────────────────────

  const handleImagesSelected = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    const paths: string[] = [];
    const isTauri = '__TAURI_INTERNALS__' in window;
    
    for (const file of files) {
      if (isTauri) {
        try {
          const buf = await file.arrayBuffer();
          const path = await invoke<string>('save_uploaded_image', { filename: file.name, data: Array.from(new Uint8Array(buf)) });
          paths.push(path);
        } catch (err) {
          showToast('error', err instanceof Error ? err.message : `Copie échouée: ${file.name}`);
          e.target.value = ''; return;
        }
      } else {
        // En mode web classique, on simule le chemin pour tester l'interface visuellement
        paths.push(URL.createObjectURL(file));
      }
    }
    
    const pages = await Promise.all(files.map((f, i) => makeEmptyPage(f, i, paths[i])));
    const now = new Date().toISOString();
    loadProject({ schema_version: '1.0.0', project_id: crypto.randomUUID(), name: `Projet ${now.slice(0, 10)}`, created_at: now, updated_at: now, settings: { source_lang: 'en', target_lang: 'fr', cache_enabled: true }, pages });
    e.target.value = '';
  };

  const handleProjectSelected = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const raw = await file.text();
    let p: Project | null = null;
    try { p = deserializeProject(raw); } catch { try { p = importLegacyMetadata(JSON.parse(raw)); } catch { p = null; } }
    if (p) loadProject(p); else showToast('error', 'Impossible de lire le project.json');
    e.target.value = '';
  };

  // ── Pipeline steps ────────────────────────────────────────────────────────

  const setStep = (step: PipelineStepId, status: StepStatus) =>
    setStepStatus((prev) => ({ ...prev, [step]: status }));

  const handleAutoRun = async () => {
    if (!project?.pages.length) { showToast('error', 'Aucun projet/page disponible.'); return; }
    try {
      resetJob();
      setStepStatus({ detect: 'running', ocr: 'idle', translate: 'idle', render: 'idle' });
      for (const page of project.pages) {
        if (!page.image_path || !isAbsoluteSystemPath(page.image_path)) {
          showToast('error', `Chemin invalide: page ${page.index + 1}`); continue;
        }
        const create = await createJob({ input_path: page.image_path, output_dir: FIXED_OUTPUT_DIR, debug: false });
        startJob(create.job_id);
        setJobStatus('running');
        const final = await pollJob(create.job_id, (snap) => {
          if (snap.logs?.length) pushLogs(snap.logs.map(l => ({ ...l, message: `[Page ${page.index + 1}] ${l.message}` })));
          setJobStatus(snap.status);
        }, 500);
        if (final.status === 'failed') throw new Error(final.error ?? `Job échoué (page ${page.index + 1})`);
        const dets = extractDetections(final.result);
        setPageBubbles(page.id, mapDetectionsToBubbles(dets));
        if (typeof final.result?.output === 'string') setPagePreviewPath(page.id, final.result.output);
      }
      setStepStatus({ detect: 'done', ocr: 'done', translate: 'done', render: 'done' });
      
      // Select the first bubble of the active page if it exists
      if (activePageId) {
        const p = project.pages.find(p => p.id === activePageId);
        if (p?.bubbles.length) {
          setActiveBubble(p.bubbles[0].id);
        }
      }
      
      showToast('success', 'Auto Run terminé.');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Erreur Auto Run';
      setJobError(msg); setJobStatus('failed');
      setStepStatus((prev) => {
        const next = { ...prev };
        (Object.keys(next) as PipelineStepId[]).forEach(k => { if (next[k] === 'running') next[k] = 'error'; });
        return next;
      });
      showToast('error', msg);
    }
  };

  const handleDetect = async () => {
    if (!activePage) { showToast('error', 'Sélectionne une page.'); return; }
    setStep('detect', 'running');
    try {
      const res = await runDetect({ image_path: activePage.image_path, classes: ['bulle', 'out_text', 'System'], debug: false });
      // The backend returns bboxes as arrays, we must map them to {x,y,w,h} objects
      const newBubbles = mapDetectionsToBubbles(res.bubbles as unknown as Array<Record<string, unknown>>);
      setPageBubbles(activePage.id, newBubbles);
      if (newBubbles.length > 0) {
        setActiveBubble(newBubbles[0].id);
      }
      setStep('detect', 'done');
      showToast('success', `Detect: ${newBubbles.length} bulles.`);
    } catch (err) { setStep('detect', 'error'); showToast('error', err instanceof Error ? err.message : 'Detect failed'); }
  };

  const handleOcr = async () => {
    if (!activePage) { showToast('error', 'Sélectionne une page.'); return; }
    setStep('ocr', 'running');
    try {
      const res = await runOcr({ 
        image_path: activePage.image_path, 
        bubbles: activePage.bubbles.map(b => ({
          ...b,
          bbox: [b.bbox.x, b.bbox.y, b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h]
        })) 
      });
      patchPageBubbles(activePage.id, (res.bubbles ?? []).map(b => ({ id: b.id, source_text: b.source_text, ocr_confidence: b.ocr_confidence, errors: b.errors })));
      setStep('ocr', 'done');
      showToast('success', 'OCR terminé.');
    } catch (err) { setStep('ocr', 'error'); showToast('error', err instanceof Error ? err.message : 'OCR failed'); }
  };

  const handleTranslate = async () => {
    if (!activePage) { showToast('error', 'Sélectionne une page.'); return; }
    setStep('translate', 'running');
    try {
      const res = await runTranslate({
        bubbles: activePage.bubbles.map(b => ({ id: b.id, source_text: b.source_override ?? b.source_text, translated_text: b.translated_override ?? b.translated_text, llm_input_index: b.llm_input_index, llm_output_index: b.llm_output_index })),
        cache_enabled: true, return_llm_debug: true,
        glossary: project?.settings?.glossary || {},
      });
      patchPageBubbles(activePage.id, (res.bubbles ?? []).map(b => ({ id: b.id, translated_text: b.translated_text, llm_input_index: b.llm_input_index, llm_output_index: b.llm_output_index, errors: b.errors })));
      setStep('translate', 'done');
      showToast('success', 'Traduction terminée.');
    } catch (err) { setStep('translate', 'error'); showToast('error', err instanceof Error ? err.message : 'Translate failed'); }
  };

  const handleRender = async () => {
    if (!activePage) { showToast('error', 'Sélectionne une page.'); return; }
    setStep('render', 'running');
    try {
      const res = await renderPagePreview({
        image_path: activePage.image_path,
        bubbles: activePage.bubbles.map(b => ({ 
          ...b, 
          source_text: b.source_override ?? b.source_text, 
          translated_text: b.translated_override ?? b.translated_text,
          bbox: [b.bbox.x, b.bbox.y, b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h]
        })),
        text_only: false, skip_inpainting: false,
      });
      setPagePreviewPath(activePage.id, res.preview_path ?? null);
      setStep('render', 'done');
      showToast('success', 'Render terminé.');
    } catch (err) { setStep('render', 'error'); showToast('error', err instanceof Error ? err.message : 'Render failed'); }
  };

  const stepHandlers: Record<PipelineStepId, () => void> = { detect: handleDetect, ocr: handleOcr, translate: handleTranslate, render: handleRender };
  const stepIcon = (s: StepStatus) => s === 'done' ? '✓' : s === 'error' ? '✕' : s === 'running' ? '⟳' : '';

  const isJobRunning = jobStatus === 'running' || jobStatus === 'queued';

  return (
    <div className="app-shell">
      {/* Hidden file inputs */}
      <input ref={imageInputRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" multiple hidden onChange={handleImagesSelected} />
      <input ref={projectInputRef} type="file" accept="application/json,.json" hidden onChange={handleProjectSelected} />

      {/* ── Topbar ── */}
      <header className="topbar">
        {/* Logo + File menu */}
        <span className="topbar-logo">
          <span>⬡</span> OMNI READ
        </span>
        <div className="topbar-divider" />
        <div className="menu-wrapper">
          <button className="menu-btn" type="button" onClick={(e) => { e.stopPropagation(); setIsFileMenuOpen(v => !v); }}>Fichier</button>
          {isFileMenuOpen ? (
            <div className="menu-dropdown" onClick={e => e.stopPropagation()}>
              <button className="menu-item" type="button" onClick={() => { setIsFileMenuOpen(false); imageInputRef.current?.click(); }}>Nouveau projet…</button>
              <button className="menu-item" type="button" onClick={() => { setIsFileMenuOpen(false); projectInputRef.current?.click(); }}>Ouvrir projet…</button>
              <button className="menu-item" type="button" onClick={() => setIsFileMenuOpen(false)}>Sauvegarder</button>
            </div>
          ) : null}
        </div>
        <div className="topbar-divider" />

        {/* Pipeline stepper */}
        <nav className="topbar-pipeline">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {i > 0 ? <span className="pipeline-arrow">›</span> : null}
              <button
                type="button"
                className={`pipeline-step ${stepStatus[step.id] === 'idle' ? '' : `is-${stepStatus[step.id]}`}`}
                onClick={stepHandlers[step.id]}
                disabled={isJobRunning}
                title={step.label}
              >
                <span className="pipeline-step-icon">{stepIcon(stepStatus[step.id]) || (i + 1)}</span>
                {step.label}
              </button>
            </div>
          ))}
        </nav>

        {/* Actions */}
        <div className="topbar-actions">
          {activePage?.preview_path && activePage.preview_path !== activePage.image_path ? (
            <button
              type="button"
              className={`btn ${showOriginal ? 'btn-ghost' : 'btn-ghost'}`}
              style={{ borderColor: showOriginal ? 'var(--accent)' : undefined, color: showOriginal ? 'var(--accent)' : undefined }}
              onClick={toggleShowOriginal}
              title="Basculer original / traduit"
            >
              {showOriginal ? '🔤 Original' : '✅ Traduit'}
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleAutoRun}
            disabled={isJobRunning || !project?.pages.length}
          >
            {isJobRunning ? '⟳ En cours…' : '▶ Auto Run'}
          </button>
          <button type="button" className="btn-icon" onClick={() => setIsGlossaryOpen(true)} title="Glossaire & Mémoire">📖</button>
          <button type="button" className="btn-icon" onClick={() => setIsBatchManagerOpen(true)} title="Batch Processing">🗂</button>
          <button type="button" className="btn-icon" onClick={() => setIsExportOpen(true)} title="Exporter">📦</button>
          <button type="button" className="btn-icon" onClick={() => setIsSettingsOpen(true)} title="Paramètres">⚙</button>
        </div>
      </header>

      {/* ── Workspace ── */}
      <main className="workspace">
        {/* Left sidebar */}
        <aside className="left-sidebar">
          <ImageListPanel />
        </aside>

        {/* Center: canvas + toolbar + logs */}
        <section className="canvas-area">
          {/* Vertical tool toolbar */}
          <div className="canvas-toolbar">
            {TOOLS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`tool-btn ${tool === t.id ? 'is-active' : ''}`}
                title={t.title}
                onClick={() => setTool(t.id)}
              >
                {t.icon}
              </button>
            ))}
          </div>

          <CanvasEditor />

          {/* Floating logs */}
          {jobLogs.length > 0 || isJobRunning ? (
            <div className="logs-panel">
              <div className="logs-header">
                <span className="logs-title">Logs pipeline</span>
                <span className={`logs-status ${isJobRunning ? 'running' : jobStatus === 'done' ? 'done' : jobStatus === 'failed' ? 'failed' : ''}`}>
                  {isJobRunning ? 'En cours' : jobStatus === 'done' ? 'Terminé' : jobStatus === 'failed' ? 'Erreur' : jobStatus}
                </span>
              </div>
              <div className="logs-scroll">
                {jobLogs.slice(-40).map((log, i) => (
                  <div
                    key={i}
                    className={`log-line ${log.level === 'ERROR' ? 'error' : log.level === 'WARNING' ? 'warning' : ''}`}
                  >
                    {log.message}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </section>

        {/* Right sidebar: inspector */}
        <aside className="right-sidebar">
          <BubbleInspectorPanel onRenderRequest={handleRender} />
        </aside>
      </main>

      {/* Toasts */}
      <div className="toast-stack">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.type}`}>{t.message}</div>
        ))}
      </div>

      {/* Settings modal */}
      {isSettingsOpen ? <SettingsPanel onClose={() => setIsSettingsOpen(false)} /> : null}
      {/* Batch modal */}
      {isBatchManagerOpen ? <BatchManager onClose={() => setIsBatchManagerOpen(false)} /> : null}
      {/* Glossary modal */}
      {isGlossaryOpen ? <GlossaryManager onClose={() => setIsGlossaryOpen(false)} /> : null}
      {/* Export modal */}
      {isExportOpen ? <ExportModal onClose={() => setIsExportOpen(false)} /> : null}
    </div>
  );
};

export default App;
