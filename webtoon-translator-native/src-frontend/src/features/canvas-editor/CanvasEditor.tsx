import { useEffect, useMemo, useRef, useState } from 'react';
import { Image as KonvaImage, Layer, Rect, Stage, Text } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCanvasStore } from './canvasStore';
import { useProjectStore } from '../project/projectStore';
const useHtmlImage = (src: string | null) => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!src) {
      setImage(null);
      return;
    }

    const htmlImage = new Image();
    htmlImage.crossOrigin = 'anonymous';
    htmlImage.onload = () => setImage(htmlImage);
    htmlImage.onerror = () => setImage(null);
    htmlImage.src = src;
  }, [src]);

  return image;
};

const CanvasEditor = () => {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 900, height: 600 });
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [lastPointer, setLastPointer] = useState<{ x: number; y: number } | null>(null);

  const viewport = useCanvasStore((state) => state.viewport);
  const setViewport = useCanvasStore((state) => state.setViewport);
  const zoomEnabled = useCanvasStore((state) => state.zoomEnabled);
  const activePageId = useCanvasStore((state) => state.activePageId);
  const activeBubbleId = useCanvasStore((state) => state.activeBubbleId);
  const setActiveBubble = useCanvasStore((state) => state.setActiveBubble);

  const project = useProjectStore((state) => state.project);
  const activePage = useMemo(
    () => project?.pages.find((page) => page.id === activePageId) ?? null,
    [activePageId, project]
  );

  useEffect(() => {
    console.log('[CanvasEditor] active page bubbles', {
      activePageId,
      pageIndex: activePage?.index,
      bubblesCount: activePage?.bubbles.length ?? 0,
      firstThree: (activePage?.bubbles ?? []).slice(0, 3).map((bubble) => ({
        id: bubble.id,
        bbox: bubble.bbox,
        source: bubble.source_text,
      })),
    });
  }, [activePage, activePageId]);

  const pageImageSrc = activePage?.preview_path ?? activePage?.image_path ?? null;
  const pageImage = useHtmlImage(pageImageSrc);

  const baseImageWidth = Math.max(1, activePage?.width ?? pageImage?.naturalWidth ?? 1000);
  const baseImageHeight = Math.max(1, activePage?.height ?? pageImage?.naturalHeight ?? 1400);

  const fitScale = useMemo(() => {
    const scaleX = stageSize.width / baseImageWidth;
    const scaleY = stageSize.height / baseImageHeight;
    const scale = Math.min(scaleX, scaleY);
    if (!Number.isFinite(scale) || scale <= 0) {
      return 1;
    }
    return scale;
  }, [baseImageHeight, baseImageWidth, stageSize.height, stageSize.width]);

  const effectiveScale = fitScale * viewport.zoom;
  const centeredX = (stageSize.width - baseImageWidth * effectiveScale) / 2;
  const centeredY = (stageSize.height - baseImageHeight * effectiveScale) / 2;
  const layerX = centeredX + viewport.pan_x;
  const layerY = centeredY + viewport.pan_y;

  useEffect(() => {
    const handleResize = () => {
      if (!wrapperRef.current) {
        return;
      }
      const bounds = wrapperRef.current.getBoundingClientRect();
      setStageSize({
        width: Math.max(320, Math.floor(bounds.width)),
        height: Math.max(240, Math.floor(bounds.height)),
      });
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        setIsSpacePressed(true);
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        setIsSpacePressed(false);
        setIsPanning(false);
        setLastPointer(null);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const cursor = useMemo(() => {
    if (isPanning) {
      return 'grabbing';
    }
    if (isSpacePressed) {
      return 'grab';
    }
    return 'default';
  }, [isPanning, isSpacePressed]);

  const onWheel = (event: KonvaEventObject<WheelEvent>) => {
    if (!zoomEnabled || !event.evt.ctrlKey) {
      return;
    }

    event.evt.preventDefault();
    const direction = event.evt.deltaY > 0 ? -1 : 1;
    const factor = 0.08;
    const nextZoom = Math.min(4, Math.max(0.25, viewport.zoom + direction * factor));
    setViewport({ zoom: Number(nextZoom.toFixed(3)) });
  };

  const onMouseDown = (event: KonvaEventObject<MouseEvent>) => {
    if (isSpacePressed) {
      setIsPanning(true);
      setLastPointer({ x: event.evt.clientX, y: event.evt.clientY });
      return;
    }

    if (event.target === event.currentTarget) {
      setActiveBubble(null);
    }
  };

  const onMouseMove = (event: KonvaEventObject<MouseEvent>) => {
    if (!isPanning || !lastPointer) {
      return;
    }

    const dx = event.evt.clientX - lastPointer.x;
    const dy = event.evt.clientY - lastPointer.y;
    setViewport({
      pan_x: viewport.pan_x + dx,
      pan_y: viewport.pan_y + dy,
    });
    setLastPointer({ x: event.evt.clientX, y: event.evt.clientY });
  };

  const onMouseUp = () => {
    setIsPanning(false);
    setLastPointer(null);
  };

  return (
    <div className="canvas-editor" ref={wrapperRef}>
      <Stage
        width={stageSize.width}
        height={stageSize.height}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        style={{ cursor }}
      >
        <Layer x={layerX} y={layerY} scaleX={effectiveScale} scaleY={effectiveScale}>
          <Rect
            x={0}
            y={0}
            width={baseImageWidth}
            height={baseImageHeight}
            fill="#171f2d"
            cornerRadius={6}
          />

          {pageImage && activePage ? (
            <KonvaImage image={pageImage} x={0} y={0} width={baseImageWidth} height={baseImageHeight} />
          ) : null}

          {activePage ? (
            activePage.bubbles.map((bubble) => (
              <Rect
                key={bubble.id}
                x={bubble.bbox.x}
                y={bubble.bbox.y}
                width={bubble.bbox.w}
                height={bubble.bbox.h}
                stroke={activeBubbleId === bubble.id ? '#2f89ff' : '#ffb266'}
                strokeWidth={activeBubbleId === bubble.id ? 3 : 2}
                dash={activeBubbleId === bubble.id ? [] : [6, 6]}
                onClick={() => setActiveBubble(bubble.id)}
              />
            ))
          ) : (
            <Text
              x={20}
              y={20}
              text="CANVAS PRINCIPAL (charger un projet pour afficher une page)"
              fill="#8ba2c9"
              fontSize={20}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
};

export default CanvasEditor;
