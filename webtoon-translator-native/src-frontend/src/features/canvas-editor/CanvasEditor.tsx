import React from 'react';
import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Image as KonvaImage, Layer, Rect, Stage, Text, Circle } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { useCanvasStore } from './canvasStore';
import { useProjectStore } from '../project/projectStore';
import type { Bubble } from '../../shared/types';

const CLASS_COLORS: Record<string, string> = {
  bulle: '#22c55e',
  system: '#a855f7',
  out_text: '#f97316',
  sfx: '#ef4444',
  'Bubble': '#22c55e',
  'Box': '#4f8ef7',
  'Outer_Text': '#f97316',
  'Small_Text': '#06b6d4',
  'Continuation': '#a855f7',
};
const getBubbleColor = (cls: string) =>
  CLASS_COLORS[cls] ?? CLASS_COLORS[cls.toLowerCase()] ?? '#4f8ef7';

const useHtmlImage = (src: string | null) => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!src) { setImage(null); return; }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => setImage(img);
    img.onerror = () => setImage(null);
    img.src = src;
  }, [src]);
  return image;
};

// Resize handle positions
const HANDLES = ['nw','n','ne','e','se','s','sw','w'] as const;
type Handle = typeof HANDLES[number];
const HANDLE_SIZE = 8;

const getHandlePos = (x: number, y: number, w: number, h: number, handle: Handle) => {
  const cx = x + w / 2, cy = y + h / 2;
  switch (handle) {
    case 'nw': return { hx: x, hy: y };
    case 'n':  return { hx: cx, hy: y };
    case 'ne': return { hx: x + w, hy: y };
    case 'e':  return { hx: x + w, hy: cy };
    case 'se': return { hx: x + w, hy: y + h };
    case 's':  return { hx: cx, hy: y + h };
    case 'sw': return { hx: x, hy: y + h };
    case 'w':  return { hx: x, hy: cy };
  }
};

const CanvasEditor = () => {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 900, height: 600 });

  // Pan state
  const [isPanning, setIsPanning] = useState(false);
  const [lastPointer, setLastPointer] = useState<{ x: number; y: number } | null>(null);

  // Draw state
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);

  // Drag/resize state
  const [dragging, setDragging] = useState<{ bubbleId: string; startX: number; startY: number; origX: number; origY: number } | null>(null);
  const [resizing, setResizing] = useState<{ bubbleId: string; handle: Handle; startX: number; startY: number; origBbox: { x: number; y: number; w: number; h: number } } | null>(null);

  const viewport = useCanvasStore((s) => s.viewport);
  const setViewport = useCanvasStore((s) => s.setViewport);
  const tool = useCanvasStore((s) => s.tool);
  const activePageId = useCanvasStore((s) => s.activePageId);
  const activeBubbleId = useCanvasStore((s) => s.activeBubbleId);
  const hoveredBubbleId = useCanvasStore((s) => s.hoveredBubbleId);
  const setActiveBubble = useCanvasStore((s) => s.setActiveBubble);
  const setHoveredBubble = useCanvasStore((s) => s.setHoveredBubble);
  const drawingRect = useCanvasStore((s) => s.drawingRect);
  const setDrawingRect = useCanvasStore((s) => s.setDrawingRect);
  const showOriginal = useCanvasStore((s) => s.showOriginal);

  const project = useProjectStore((s) => s.project);
  const setPageBubbles = useProjectStore((s) => s.setPageBubbles);
  const patchPageBubbles = useProjectStore((s) => s.patchPageBubbles);

  const activePage = useMemo(
    () => project?.pages.find((p) => p.id === activePageId) ?? null,
    [activePageId, project]
  );

  // Image source: original vs preview toggle
  const pageImageSrc = useMemo(() => {
    if (showOriginal) return activePage?.image_path ?? null;
    return activePage?.preview_path ?? activePage?.image_path ?? null;
  }, [activePage, showOriginal]);

  const pageImage = useHtmlImage(pageImageSrc);

  const baseW = Math.max(1, activePage?.width ?? pageImage?.naturalWidth ?? 1000);
  const baseH = Math.max(1, activePage?.height ?? pageImage?.naturalHeight ?? 1400);

  const fitScale = useMemo(() => {
    const sx = stageSize.width / baseW;
    const sy = stageSize.height / baseH;
    const s = Math.min(sx, sy, 1); // never upscale beyond 1x on fit
    return Number.isFinite(s) && s > 0 ? s : 1;
  }, [baseH, baseW, stageSize.height, stageSize.width]);

  const effectiveScale = fitScale * viewport.zoom;
  const maxPanY = Math.max(0, (baseH * effectiveScale - stageSize.height) / 2);
  // Restrict pan_y to bounds if zooming out
  const clampedPanY = Math.min(maxPanY, Math.max(-maxPanY, viewport.pan_y));
  
  const layerX = (stageSize.width - baseW * effectiveScale) / 2 + viewport.pan_x;
  const layerY = (stageSize.height - baseH * effectiveScale) / 2 + clampedPanY;

  // Stage coords → image coords
  const toImageCoords = useCallback((stageX: number, stageY: number) => ({
    x: (stageX - layerX) / effectiveScale,
    y: (stageY - layerY) / effectiveScale,
  }), [layerX, layerY, effectiveScale]);

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setStageSize({ width: Math.max(320, Math.floor(r.width)), height: Math.max(240, Math.floor(r.height)) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Delete' || e.code === 'Backspace') {
        if (activeBubbleId && activePage && document.activeElement?.tagName !== 'TEXTAREA' && document.activeElement?.tagName !== 'INPUT') {
          setPageBubbles(activePage.id, activePage.bubbles.filter(b => b.id !== activeBubbleId));
          setActiveBubble(null);
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeBubbleId, activePage, setActiveBubble, setPageBubbles]);

  // Wheel zoom
  const onWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const factor = 0.2; // Increased zoom step
    // Allow zooming up to 50x
    const nextZoom = Math.min(50, Math.max(0.1, viewport.zoom + direction * factor * viewport.zoom));
    setViewport({ zoom: Number(nextZoom.toFixed(3)) });
  };

  // --- MOUSE DOWN ---
  const onStageMouseDown = (e: KonvaEventObject<MouseEvent>) => {
    const pos = e.target.getStage()!.getPointerPosition()!;
    const imgPos = toImageCoords(pos.x, pos.y);

    if (tool === 'pan' || e.evt.button === 1) {
      setIsPanning(true);
      setLastPointer({ x: pos.x, y: pos.y });
      return;
    }

    if (tool === 'draw') {
      setDrawStart(imgPos);
      setDrawingRect({ x: imgPos.x, y: imgPos.y, w: 0, h: 0 });
      setActiveBubble(null);
      return;
    }

    if (tool === 'select') {
      // Deselect if clicked on stage background
      if (e.target === e.currentTarget) {
        setActiveBubble(null);
      }
    }
  };

  // --- MOUSE MOVE ---
  const onStageMouseMove = (e: KonvaEventObject<MouseEvent>) => {
    const pos = e.target.getStage()!.getPointerPosition()!;
    const imgPos = toImageCoords(pos.x, pos.y);

    // Pan
    if (isPanning && lastPointer) {
      setViewport({ pan_x: viewport.pan_x + pos.x - lastPointer.x, pan_y: viewport.pan_y + pos.y - lastPointer.y });
      setLastPointer({ x: pos.x, y: pos.y });
      return;
    }

    // Draw
    if (tool === 'draw' && drawStart) {
      const x = Math.min(drawStart.x, imgPos.x);
      const y = Math.min(drawStart.y, imgPos.y);
      const w = Math.abs(imgPos.x - drawStart.x);
      const h = Math.abs(imgPos.y - drawStart.y);
      setDrawingRect({ x, y, w, h });
      return;
    }

    // Move bubble
    if (dragging && activePage) {
      const dx = (pos.x - dragging.startX) / effectiveScale;
      const dy = (pos.y - dragging.startY) / effectiveScale;
      patchPageBubbles(activePage.id, [{
        id: dragging.bubbleId,
        bbox: {
          x: Math.max(0, dragging.origX + dx),
          y: Math.max(0, dragging.origY + dy),
          w: activePage.bubbles.find(b => b.id === dragging.bubbleId)?.bbox.w ?? 100,
          h: activePage.bubbles.find(b => b.id === dragging.bubbleId)?.bbox.h ?? 60,
        },
      }]);
      return;
    }

    // Resize
    if (resizing && activePage) {
      const dx = (pos.x - resizing.startX) / effectiveScale;
      const dy = (pos.y - resizing.startY) / effectiveScale;
      const ob = resizing.origBbox;
      let { x, y, w, h } = ob;

      switch (resizing.handle) {
        case 'se': w = Math.max(20, ob.w + dx); h = Math.max(20, ob.h + dy); break;
        case 'sw': x = Math.min(ob.x + ob.w - 20, ob.x + dx); w = Math.max(20, ob.w - dx); h = Math.max(20, ob.h + dy); break;
        case 'ne': w = Math.max(20, ob.w + dx); y = Math.min(ob.y + ob.h - 20, ob.y + dy); h = Math.max(20, ob.h - dy); break;
        case 'nw': x = Math.min(ob.x + ob.w - 20, ob.x + dx); y = Math.min(ob.y + ob.h - 20, ob.y + dy); w = Math.max(20, ob.w - dx); h = Math.max(20, ob.h - dy); break;
        case 'n': y = Math.min(ob.y + ob.h - 20, ob.y + dy); h = Math.max(20, ob.h - dy); break;
        case 's': h = Math.max(20, ob.h + dy); break;
        case 'e': w = Math.max(20, ob.w + dx); break;
        case 'w': x = Math.min(ob.x + ob.w - 20, ob.x + dx); w = Math.max(20, ob.w - dx); break;
      }
      patchPageBubbles(activePage.id, [{ id: resizing.bubbleId, bbox: { x, y, w, h } }]);
    }
  };

  // --- MOUSE UP ---
  const onStageMouseUp = (e: KonvaEventObject<MouseEvent>) => {
    setIsPanning(false);
    setLastPointer(null);
    setDragging(null);
    setResizing(null);

    if (tool === 'draw' && drawStart && drawingRect && activePage) {
      if (drawingRect.w > 10 && drawingRect.h > 10) {
        const newBubble: Bubble = {
          id: crypto.randomUUID(),
          bbox: drawingRect,
          class: 'bulle',
          source_text: '',
          translated_text: '',
          source_override: null,
          translated_override: null,
          llm_input_index: null,
          llm_output_index: null,
          detection_confidence: null,
          ocr_confidence: null,
          text_style: { font_family: 'Anime Ace', font_size: 24, align: 'center', color: '#000000' },
          mask_strokes: [],
          errors: [],
        };
        setPageBubbles(activePage.id, [...activePage.bubbles, newBubble]);
        setActiveBubble(newBubble.id);
      }
      setDrawStart(null);
      setDrawingRect(null);
    }
  };

  const cursor = useMemo(() => {
    if (isPanning || dragging || resizing) return 'grabbing';
    if (tool === 'pan') return 'grab';
    if (tool === 'draw') return 'crosshair';
    if (tool === 'delete') return 'not-allowed';
    return 'default';
  }, [isPanning, tool, dragging, resizing]);

  return (
    <div className="canvas-editor" ref={wrapperRef} style={{ position: 'relative' }}>
      {maxPanY > 0 && (
        <input
          type="range"
          min={-maxPanY}
          max={maxPanY}
          value={-clampedPanY}
          onChange={(e) => setViewport({ pan_y: -Number(e.target.value) })}
          style={{
            position: 'absolute',
            right: 12,
            top: 24,
            bottom: 24,
            width: 16,
            height: 'calc(100% - 48px)',
            appearance: 'slider-vertical',
            WebkitAppearance: 'slider-vertical',
            writingMode: 'bt-lr',
            zIndex: 100,
            cursor: 'ns-resize',
          }}
          title="Faire défiler l'image verticalement"
        />
      )}
      <Stage
        width={stageSize.width}
        height={stageSize.height}
        onWheel={onWheel}
        onMouseDown={onStageMouseDown}
        onMouseMove={onStageMouseMove}
        onMouseUp={onStageMouseUp}
        onMouseLeave={onStageMouseUp}
        style={{ cursor, display: 'block' }}
      >
        <Layer x={layerX} y={layerY} scaleX={effectiveScale} scaleY={effectiveScale}>
          {/* Background */}
          <Rect x={0} y={0} width={baseW} height={baseH} fill="#090c12" />

          {/* Page image */}
          {pageImage && activePage ? (
            <KonvaImage image={pageImage} x={0} y={0} width={baseW} height={baseH} />
          ) : null}

          {/* Empty state */}
          {!activePage ? (
            <Text
              x={baseW / 2 - 200}
              y={baseH / 2 - 30}
              width={400}
              align="center"
              text="Chargez un projet ou sélectionnez une image"
              fill="#2a3a52"
              fontSize={16}
            />
          ) : null}

          {/* Bboxes */}
          {activePage?.bubbles.map((bubble) => {
            const isActive = activeBubbleId === bubble.id;
            const isHovered = hoveredBubbleId === bubble.id;
            const color = getBubbleColor(bubble.class);
            return (
              <React.Fragment key={bubble.id}>
                <Rect
                  x={bubble.bbox.x}
                  y={bubble.bbox.y}
                  width={bubble.bbox.w}
                  height={bubble.bbox.h}
                  stroke={isActive ? color : isHovered ? color : color}
                  strokeWidth={isActive ? 4 / effectiveScale : isHovered ? 3 / effectiveScale : 2.5 / effectiveScale}
                  fill={isActive ? `${color}25` : isHovered ? `${color}15` : `${color}00`}
                  dash={isActive ? undefined : [6 / effectiveScale, 4 / effectiveScale]}
                  opacity={isActive ? 1 : 0.6}
                  onMouseEnter={() => setHoveredBubble(bubble.id)}
                  onMouseLeave={() => setHoveredBubble(null)}
                  onClick={() => {
                    if (tool === 'delete' && activePage) {
                      setPageBubbles(activePage.id, activePage.bubbles.filter(b => b.id !== bubble.id));
                      setActiveBubble(null);
                    } else {
                      setActiveBubble(bubble.id);
                    }
                  }}
                  onMouseDown={(e) => {
                    if (tool !== 'select') return;
                    e.cancelBubble = true;
                    setActiveBubble(bubble.id);
                    const stage = e.target.getStage()!;
                    const pos = stage.getPointerPosition()!;
                    setDragging({ bubbleId: bubble.id, startX: pos.x, startY: pos.y, origX: bubble.bbox.x, origY: bubble.bbox.y });
                  }}
                />
                {/* Resize handles - only for active bubble in select mode */}
                {isActive && tool === 'select' && HANDLES.map((handle) => {
                  const { hx, hy } = getHandlePos(bubble.bbox.x, bubble.bbox.y, bubble.bbox.w, bubble.bbox.h, handle);
                  const hs = HANDLE_SIZE / effectiveScale;
                  return (
                    <Rect
                      key={handle}
                      x={hx - hs / 2}
                      y={hy - hs / 2}
                      width={hs}
                      height={hs}
                      fill="white"
                      stroke={color}
                      strokeWidth={1.5 / effectiveScale}
                      onMouseDown={(e) => {
                        e.cancelBubble = true;
                        const stage = e.target.getStage()!;
                        const pos = stage.getPointerPosition()!;
                        setResizing({ bubbleId: bubble.id, handle, startX: pos.x, startY: pos.y, origBbox: { ...bubble.bbox } });
                      }}
                    />
                  );
                })}
                {/* Bubble number label (only on hover/active) */}
                {isActive || isHovered ? (
                  <Text
                    x={bubble.bbox.x + 3 / effectiveScale}
                    y={bubble.bbox.y - 16 / effectiveScale}
                    text={bubble.class}
                    fill={color}
                    fontSize={12 / effectiveScale}
                    fontStyle="bold"
                  />
                ) : null}
              </React.Fragment>
            );
          })}

          {/* Drawing preview */}
          {drawingRect && drawingRect.w > 2 && drawingRect.h > 2 ? (
            <Rect
              x={drawingRect.x}
              y={drawingRect.y}
              width={drawingRect.w}
              height={drawingRect.h}
              stroke="#4f8ef7"
              strokeWidth={2 / effectiveScale}
              fill="rgba(79,142,247,0.08)"
              dash={[6 / effectiveScale, 4 / effectiveScale]}
            />
          ) : null}
        </Layer>
      </Stage>
    </div>
  );
};

export default CanvasEditor;
