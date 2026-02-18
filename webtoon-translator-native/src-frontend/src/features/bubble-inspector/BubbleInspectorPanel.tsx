import { useState } from 'react';
import { useCanvasStore } from '../canvas-editor/canvasStore';
import LlmMappingPanel from '../llm-mapping-debug/LlmMappingPanel';
import { useProjectStore } from '../project/projectStore';
import { renderPagePreview } from '../render/renderService';

type TabKey = 'texte' | 'style' | 'debug';

const BubbleInspectorPanel = () => {
  const [activeTab, setActiveTab] = useState<TabKey>('texte');
  const [isRerendering, setIsRerendering] = useState(false);
  const project = useProjectStore((state) => state.project);
  const updateBubbleOverrides = useProjectStore((state) => state.updateBubbleOverrides);
  const setPagePreviewPath = useProjectStore((state) => state.setPagePreviewPath);
  const patchPageBubbles = useProjectStore((state) => state.patchPageBubbles);
  const activePageId = useCanvasStore((state) => state.activePageId);
  const activeBubbleId = useCanvasStore((state) => state.activeBubbleId);
  const setActiveBubble = useCanvasStore((state) => state.setActiveBubble);

  const activePage = project?.pages.find((page) => page.id === activePageId) ?? null;
  const activeBubble = activePage?.bubbles.find((bubble) => bubble.id === activeBubbleId) ?? null;

  if (!activePage || !activeBubble) {
    return null;
  }

  const handleRerender = async () => {
    try {
      setIsRerendering(true);
      const response = await renderPagePreview({
        image_path: activePage.image_path,
        bubbles: activePage.bubbles.map((bubble) => ({
          ...bubble,
          source_text: bubble.source_override ?? bubble.source_text,
          translated_text: bubble.translated_override ?? bubble.translated_text,
        })),
        text_only: true,
        skip_inpainting: true,
      });

      setPagePreviewPath(activePage.id, response.preview_path ?? null);
      if (response.errors?.length) {
        patchPageBubbles(
          activePage.id,
          response.errors
            .filter((error) => !!error.bubble_id)
            .map((error) => ({
              id: error.bubble_id as string,
              errors: [{ code: error.code, message: error.message }],
            }))
        );
      }
    } finally {
      setIsRerendering(false);
    }
  };

  return (
    <div className="properties-panel">
      <header className="properties-header">
        <h3>Props Bulle</h3>
        <button type="button" className="close-btn" onClick={() => setActiveBubble(null)}>
          X
        </button>
      </header>

      <p className="props-subtitle">Bulle active: {activeBubble.id}</p>

      <div className="bubble-meta-grid">
        <span>Class</span>
        <strong>{activeBubble.class}</strong>
        <span>Detection conf</span>
        <strong>{activeBubble.detection_confidence ?? '-'}</strong>
        <span>OCR conf</span>
        <strong>{activeBubble.ocr_confidence ?? '-'}</strong>
      </div>

      <div className="tabs-row">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'texte' ? 'is-active' : ''}`}
          onClick={() => setActiveTab('texte')}
        >
          Texte
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === 'style' ? 'is-active' : ''}`}
          onClick={() => setActiveTab('style')}
        >
          Style
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === 'debug' ? 'is-active' : ''}`}
          onClick={() => setActiveTab('debug')}
        >
          Debug Mapping
        </button>
      </div>

      <section className="tab-content">
        {activeTab === 'texte' ? (
          <div className="placeholder-block">
            <p>Source Text</p>
            <textarea readOnly value={activeBubble.source_text} />
            <p>Source Override</p>
            <textarea
              value={activeBubble.source_override ?? ''}
              onChange={(event) =>
                updateBubbleOverrides(activePage.id, activeBubble.id, {
                  source_override: event.target.value || null,
                })
              }
            />
            <p>Translated Text</p>
            <textarea readOnly value={activeBubble.translated_text} />
            <p>Translated Override</p>
            <textarea
              value={activeBubble.translated_override ?? ''}
              onChange={(event) =>
                updateBubbleOverrides(activePage.id, activeBubble.id, {
                  translated_override: event.target.value || null,
                })
              }
            />

            <button className="action-btn" type="button" onClick={handleRerender} disabled={isRerendering}>
              {isRerendering ? 'Rerender...' : 'Rerender'}
            </button>

            {activeBubble.errors?.length ? (
              <div className="bubble-errors">
                <p>Erreurs bulle</p>
                {activeBubble.errors.map((error, idx) => (
                  <div key={`${error.code}-${idx}`} className="bubble-error-item">
                    [{error.code}] {error.message}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {activeTab === 'style' ? (
          <div className="placeholder-block">
            <p>Style controls (taille, align, police, couleur)</p>
          </div>
        ) : null}

        {activeTab === 'debug' ? <LlmMappingPanel /> : null}
      </section>
    </div>
  );
};

export default BubbleInspectorPanel;
