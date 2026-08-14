import { useState } from 'react';
import { useCanvasStore } from '../canvas-editor/canvasStore';
import { useProjectStore } from '../project/projectStore';
import { renderPagePreview } from '../render/renderService';
import LlmMappingPanel from '../llm-mapping-debug/LlmMappingPanel';

type Tab = 'texte' | 'style' | 'debug';

const CLASS_LABEL: Record<string, string> = {
  bulle: 'bulle', system: 'system', out_text: 'out_text', sfx: 'sfx',
  Bubble: 'bulle', Box: 'bulle', Outer_Text: 'out_text', Small_Text: 'bulle',
};

const BubbleInspectorPanel = ({ onRenderRequest }: { onRenderRequest?: () => void }) => {
  const [tab, setTab] = useState<Tab>('texte');
  const [isRerendering, setIsRerendering] = useState(false);

  const project = useProjectStore((s) => s.project);
  const updateBubbleOverrides = useProjectStore((s) => s.updateBubbleOverrides);
  const setPagePreviewPath = useProjectStore((s) => s.setPagePreviewPath);
  const patchPageBubbles = useProjectStore((s) => s.patchPageBubbles);
  const setPageBubbles = useProjectStore((s) => s.setPageBubbles);

  const activePageId = useCanvasStore((s) => s.activePageId);
  const activeBubbleId = useCanvasStore((s) => s.activeBubbleId);
  const setActiveBubble = useCanvasStore((s) => s.setActiveBubble);

  const activePage = project?.pages.find((p) => p.id === activePageId) ?? null;
  const activeBubble = activePage?.bubbles.find((b) => b.id === activeBubbleId) ?? null;

  if (!activeBubble || !activePage) {
    return (
      <div className="inspector-panel" style={{ overflowY: 'auto' }}>
        <div style={{ padding: '24px 16px', textAlign: 'center', opacity: 0.5, flexShrink: 0 }}>
          <span className="inspector-empty-icon">⬚</span>
          <p>Sélectionne une bulle</p>
          <p style={{ fontSize: 10 }}>Clique sur une bbox dans le canvas pour l'éditer</p>
        </div>
        <hr style={{ borderColor: 'var(--border)', margin: '0 16px 16px 16px', flexShrink: 0 }} />
        <div style={{ padding: '0 16px 16px 16px', flexShrink: 0 }}>
          {activePage && <LlmMappingPanel />}
        </div>
      </div>
    );
  }

  const badgeClass = CLASS_LABEL[activeBubble.class] ?? 'bulle';
  const confDet = activeBubble.detection_confidence != null
    ? `${(activeBubble.detection_confidence * 100).toFixed(0)}%` : '—';
  const confOcr = activeBubble.ocr_confidence != null
    ? `${(activeBubble.ocr_confidence * 100).toFixed(0)}%` : '—';

  const handleRerender = async () => {
    try {
      setIsRerendering(true);
      const response = await renderPagePreview({
        image_path: activePage.image_path,
        bubbles: activePage.bubbles.map((b) => ({
          ...b,
          source_text: b.source_override ?? b.source_text,
          translated_text: b.translated_override ?? b.translated_text,
          bbox: [b.bbox.x, b.bbox.y, b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h]
        })),
        text_only: true,
        skip_inpainting: true,
      });
      setPagePreviewPath(activePage.id, response.preview_path ?? null);
      if (response.errors?.length) {
        patchPageBubbles(activePage.id,
          response.errors.filter(e => !!e.bubble_id).map(e => ({
            id: e.bubble_id as string,
            errors: [{ code: e.code, message: e.message }],
          }))
        );
      }
    } finally {
      setIsRerendering(false);
    }
  };

  const handleDelete = () => {
    setPageBubbles(activePage.id, activePage.bubbles.filter(b => b.id !== activeBubble.id));
    setActiveBubble(null);
  };

  return (
    <div className="inspector-panel">
      {/* Header */}
      <div className="inspector-header">
        <h3 className="inspector-title">Bulle</h3>
        <span className={`class-badge ${badgeClass}`}>{activeBubble.class}</span>
        <button
          type="button"
          className="btn-icon"
          style={{ color: 'var(--danger)', marginLeft: 4 }}
          onClick={handleDelete}
          title="Supprimer la bulle"
        >
          🗑
        </button>
        <button
          type="button"
          className="btn-icon"
          onClick={() => setActiveBubble(null)}
          title="Fermer"
        >
          ✕
        </button>
      </div>

      {/* Confidence row */}
      <div className="confidence-row" style={{ padding: '8px 12px 0' }}>
        <div className="confidence-item">
          <div className="confidence-label">DÉTECTION</div>
          <div className="confidence-value">{confDet}</div>
        </div>
        <div className="confidence-item">
          <div className="confidence-label">OCR</div>
          <div className="confidence-value">{confOcr}</div>
        </div>
        <div className="confidence-item">
          <div className="confidence-label">BBOX</div>
          <div className="confidence-value" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
            {Math.round(activeBubble.bbox.x)},{Math.round(activeBubble.bbox.y)} {Math.round(activeBubble.bbox.w)}×{Math.round(activeBubble.bbox.h)}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="inspector-tabs">
        {(['texte', 'style', 'debug'] as Tab[]).map(t => (
          <button
            key={t}
            type="button"
            className={`inspector-tab ${tab === t ? 'is-active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t === 'texte' ? 'Texte' : t === 'style' ? 'Style' : 'Debug'}
          </button>
        ))}
      </div>

      {/* Body */}
      <div className="inspector-body">
        {tab === 'texte' ? (
          <>
            <div className="field-group" style={{ marginBottom: 12 }}>
              <button 
                type="button" 
                className="btn btn-primary" 
                style={{ width: '100%' }}
                onClick={onRenderRequest}
                title="Mettre à jour l'image avec ce texte"
              >
                🪄 Appliquer sur l'image
              </button>
            </div>
            <div className="field-group">
              <label className="field-label">Texte source (OCR)</label>
              <textarea className="field-readonly" readOnly value={activeBubble.source_text} rows={3} />
            </div>
            <div className="field-group">
              <label className="field-label">Override source</label>
              <textarea
                className="field-input"
                rows={2}
                placeholder="Laisser vide pour utiliser l'OCR"
                value={activeBubble.source_override ?? ''}
                onChange={(e) => updateBubbleOverrides(activePage.id, activeBubble.id, {
                  source_override: e.target.value || null,
                })}
              />
            </div>
            <div className="field-group">
              <label className="field-label">Traduction (pipeline)</label>
              <textarea className="field-readonly" readOnly value={activeBubble.translated_text} rows={3} />
            </div>
            <div className="field-group">
              <label className="field-label">Override traduction</label>
              <textarea
                className="field-input"
                rows={2}
                placeholder="Laisser vide pour utiliser le pipeline"
                value={activeBubble.translated_override ?? ''}
                onChange={(e) => updateBubbleOverrides(activePage.id, activeBubble.id, {
                  translated_override: e.target.value || null,
                })}
              />
            </div>
            {activeBubble.errors?.length ? (
              <div style={{ padding: '8px', borderRadius: 6, border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.06)', marginBottom: 8 }}>
                <div style={{ fontSize: 11, color: 'var(--danger)', fontWeight: 600, marginBottom: 4 }}>Erreurs</div>
                {activeBubble.errors.map((err, i) => (
                  <div key={i} style={{ fontSize: 11, color: '#fca5a5' }}>[{err.code}] {err.message}</div>
                ))}
              </div>
            ) : null}
            <button
              type="button"
              className="rerender-btn"
              onClick={handleRerender}
              disabled={isRerendering}
            >
              {isRerendering ? (
                <><span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⟳</span> Rendu en cours…</>
              ) : (
                <><span>↻</span> Rerender (texte seul)</>
              )}
            </button>
          </>
        ) : null}

        {tab === 'style' ? (
          <>
            <div className="field-group" style={{ marginBottom: 12 }}>
              <button 
                type="button" 
                className="btn btn-primary" 
                style={{ width: '100%' }}
                onClick={onRenderRequest}
                title="Mettre à jour l'image avec ces styles"
              >
                🪄 Appliquer sur l'image
              </button>
            </div>
            <div className="field-group">
              <label className="field-label">Police</label>
              <select
                className="style-select"
                value={activeBubble.text_style.font_family}
                onChange={(e) => patchPageBubbles(activePage.id, [{
                  id: activeBubble.id,
                  text_style: { ...activeBubble.text_style, font_family: e.target.value },
                }])}
              >
                {['Anime Ace', 'Inter', 'Arial', 'Comic Sans MS', 'Segoe UI'].map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Taille</label>
                <select
                  className="style-select"
                  value={activeBubble.text_style.font_size}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, font_size: Number(e.target.value) },
                  }])}
                >
                  {[10,12,14,16,18,20,22,24,28,32,36,40,48].map(s => (
                    <option key={s} value={s}>{s}px</option>
                  ))}
                </select>
              </div>
              <div className="field-group">
                <label className="field-label">Alignement</label>
                <select
                  className="style-select"
                  value={activeBubble.text_style.align}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, align: e.target.value as 'left'|'center'|'right' },
                  }])}
                >
                  <option value="left">Gauche</option>
                  <option value="center">Centre</option>
                  <option value="right">Droite</option>
                </select>
              </div>
            </div>
            <div className="field-group">
              <label className="field-label">Couleur texte</label>
              <div className="style-color-row">
                <input
                  type="color"
                  className="style-color-input"
                  value={activeBubble.text_style.color}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, color: e.target.value },
                  }])}
                />
                <input
                  type="text"
                  className="style-color-hex field-input"
                  value={activeBubble.text_style.color}
                  onChange={(e) => {
                    if (/^#[0-9a-fA-F]{6}$/.test(e.target.value)) {
                      patchPageBubbles(activePage.id, [{
                        id: activeBubble.id,
                        text_style: { ...activeBubble.text_style, color: e.target.value },
                      }]);
                    }
                  }}
                />
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Couleur contour</label>
              <div className="style-color-row">
                <input
                  type="color"
                  className="style-color-input"
                  value={activeBubble.text_style.stroke_color || '#ffffff'}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, stroke_color: e.target.value },
                  }])}
                />
                <input
                  type="text"
                  className="style-color-hex field-input"
                  placeholder="#ffffff"
                  value={activeBubble.text_style.stroke_color || ''}
                  onChange={(e) => {
                    if (/^#[0-9a-fA-F]{6}$/.test(e.target.value) || e.target.value === '') {
                      patchPageBubbles(activePage.id, [{
                        id: activeBubble.id,
                        text_style: { ...activeBubble.text_style, stroke_color: e.target.value || undefined },
                      }]);
                    }
                  }}
                />
              </div>
            </div>

            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Épaisseur contour</label>
                <input
                  type="number"
                  className="field-input"
                  min="0" max="10" step="1"
                  value={activeBubble.text_style.stroke_width || 0}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, stroke_width: Number(e.target.value) },
                  }])}
                />
              </div>
              <div className="field-group">
                <label className="field-label">Angle (°)</label>
                <input
                  type="number"
                  className="field-input"
                  min="-180" max="180" step="1"
                  value={activeBubble.text_style.angle || 0}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, angle: Number(e.target.value) },
                  }])}
                />
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Couleur de fond (Optionnel)</label>
              <div className="style-color-row">
                <input
                  type="color"
                  className="style-color-input"
                  value={activeBubble.text_style.bg_color || '#ffffff'}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    text_style: { ...activeBubble.text_style, bg_color: e.target.value },
                  }])}
                />
                <input
                  type="text"
                  className="style-color-hex field-input"
                  placeholder="#ffffff (Vide = Transparent)"
                  value={activeBubble.text_style.bg_color || ''}
                  onChange={(e) => {
                    if (/^#[0-9a-fA-F]{6}$/.test(e.target.value) || e.target.value === '') {
                      patchPageBubbles(activePage.id, [{
                        id: activeBubble.id,
                        text_style: { ...activeBubble.text_style, bg_color: e.target.value || undefined },
                      }]);
                    }
                  }}
                />
              </div>
            </div>
          </>
        ) : null}

        {tab === 'debug' ? (
          <>
            <div className="field-row">
              <div className="field-group">
                <label className="field-label">Index LLM input</label>
                <input
                  type="number"
                  className="field-input"
                  style={{ width: '100%', minHeight: 'auto', padding: '6px 8px' }}
                  value={activeBubble.llm_input_index ?? ''}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    llm_input_index: e.target.value ? Number(e.target.value) : null,
                  }])}
                />
              </div>
              <div className="field-group">
                <label className="field-label">Index LLM output</label>
                <input
                  type="number"
                  className="field-input"
                  style={{ width: '100%', minHeight: 'auto', padding: '6px 8px' }}
                  value={activeBubble.llm_output_index ?? ''}
                  onChange={(e) => patchPageBubbles(activePage.id, [{
                    id: activeBubble.id,
                    llm_output_index: e.target.value ? Number(e.target.value) : null,
                  }])}
                />
              </div>
            </div>
            <div className="field-group">
              <label className="field-label">UUID bulle</label>
              <input
                type="text"
                readOnly
                className="field-readonly"
                style={{ width: '100%', minHeight: 'auto', padding: '5px 8px', fontSize: 10, fontFamily: 'monospace' }}
                value={activeBubble.id}
              />
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};

export default BubbleInspectorPanel;
