import { useMemo } from 'react';
import { useImageListStore } from './imageListStore';
import { useCanvasStore } from '../canvas-editor/canvasStore';

const ImageListPanel = () => {
  const pages = useImageListStore((state) => state.pages);
  const selectedPageId = useImageListStore((state) => state.selectedPageId);
  const selectPage = useImageListStore((state) => state.selectPage);
  const setActivePage = useCanvasStore((state) => state.setActivePage);

  const displayPages = useMemo(
    () =>
      pages.map((page, index) => ({
        id: page.id,
        index,
        src: page.preview_path ?? page.image_path,
      })),
    [pages]
  );

  const activePageId = useCanvasStore((state) => state.activePageId);
  const setActiveBubble = useCanvasStore((state) => state.setActiveBubble);
  const activeId = activePageId ?? selectedPageId ?? displayPages[0]?.id ?? null;

  return (
    <div className="image-list-panel">
      <h2 className="panel-title">Images</h2>

      <div className="thumb-list">
        {!displayPages.length ? <p className="empty-text">Aucune image chargée</p> : null}

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
                <img
                  src={page.src}
                  alt={`Page ${page.index + 1}`}
                  className="thumb-image"
                />
              ) : null}
            </div>
            <span className="thumb-label">Page {page.index + 1}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ImageListPanel;
