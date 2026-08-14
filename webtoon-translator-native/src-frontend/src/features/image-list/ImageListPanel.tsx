import { useMemo } from 'react';
import { useImageListStore } from './imageListStore';
import { useCanvasStore } from '../canvas-editor/canvasStore';
import { useProjectStore } from '../project/projectStore';

const ImageListPanel = () => {
  const pages = useImageListStore((s) => s.pages);
  const selectedPageId = useImageListStore((s) => s.selectedPageId);
  const selectPage = useImageListStore((s) => s.selectPage);
  const activePageId = useCanvasStore((s) => s.activePageId);
  const setActivePage = useCanvasStore((s) => s.setActivePage);
  const setActiveBubble = useCanvasStore((s) => s.setActiveBubble);
  const project = useProjectStore((s) => s.project);

  const activeId = activePageId ?? selectedPageId ?? null;

  const displayPages = useMemo(() =>
    pages.map((page) => ({
      id: page.id,
      index: page.index,
      src: page.preview_path ?? page.image_path,
      bubbleCount: project?.pages.find(p => p.id === page.id)?.bubbles.length ?? 0,
    })),
    [pages, project]
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div className="sidebar-header">
        <h2 className="sidebar-title">Pages</h2>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{displayPages.length}</span>
      </div>

      <div className="thumb-list">
        {!displayPages.length ? (
          <p className="thumb-empty">Aucune image chargée</p>
        ) : null}

        {displayPages.map((page) => (
          <button
            key={page.id}
            className={`thumb-item ${activeId === page.id ? 'is-active' : ''}`}
            type="button"
            onClick={() => {
              selectPage(page.id);
              setActivePage(page.id);
              setActiveBubble(null);
            }}
          >
            <div className="thumb-preview">
              {page.src ? (
                <img src={page.src} alt={`Page ${page.index + 1}`} className="thumb-image" />
              ) : (
                <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>Aucun aperçu</span>
              )}
            </div>
            <div className="thumb-footer">
              <span className="thumb-label">Page {page.index + 1}</span>
              {page.bubbleCount > 0 ? (
                <span className="thumb-badge">{page.bubbleCount}</span>
              ) : null}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ImageListPanel;
