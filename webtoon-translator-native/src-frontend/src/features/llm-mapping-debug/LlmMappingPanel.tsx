import { useState } from 'react';
import { useCanvasStore } from '../canvas-editor/canvasStore';
import { useProjectStore } from '../project/projectStore';

const LlmMappingPanel = () => {
  const activePageId = useCanvasStore((s) => s.activePageId);
  const project = useProjectStore((s) => s.project);
  const patchPageBubbles = useProjectStore((s) => s.patchPageBubbles);
  
  const activePage = project?.pages.find((p) => p.id === activePageId);
  const bubbles = activePage?.bubbles || [];

  const [draggedIdx, setDraggedIdx] = useState<number | null>(null);

  if (!activePage) return <div className="p-4 text-gray-500">Sélectionnez une page pour remapper.</div>;

  const handleDragStart = (e: React.DragEvent, idx: number) => {
    setDraggedIdx(idx);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, targetIdx: number) => {
    e.preventDefault();
    if (draggedIdx === null || draggedIdx === targetIdx) return;
    
    // Swap translated text between draggedIdx and targetIdx
    const sourceBubble = bubbles[draggedIdx];
    const targetBubble = bubbles[targetIdx];
    
    patchPageBubbles(activePage.id, [
      { id: sourceBubble.id, translated_text: targetBubble.translated_text },
      { id: targetBubble.id, translated_text: sourceBubble.translated_text }
    ]);
    
    setDraggedIdx(null);
  };

  return (
    <div className="flex flex-col gap-4">
      <h3 className="font-bold text-sm text-gray-300">Remap Traductions</h3>
      <p className="text-xs text-gray-400">Glissez-déposez la colonne Traduction pour intervertir les textes si l'IA s'est décalée.</p>
      
      <div className="flex flex-col gap-2">
        {bubbles.map((b, i) => (
          <div key={b.id} className="flex gap-2 text-xs border border-gray-700 bg-gray-800 rounded overflow-hidden">
            <div className="flex-1 p-2 bg-gray-900 border-r border-gray-700">
              <div className="text-gray-500 mb-1">Source ({i})</div>
              <div>{b.source_override || b.source_text}</div>
            </div>
            
            <div 
              className={`flex-1 p-2 cursor-grab active:cursor-grabbing ${draggedIdx === i ? 'opacity-50 border-accent' : ''}`}
              draggable
              onDragStart={(e) => handleDragStart(e, i)}
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, i)}
            >
              <div className="text-accent mb-1">Traduction ↕</div>
              <div>{b.translated_override || b.translated_text}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LlmMappingPanel;
