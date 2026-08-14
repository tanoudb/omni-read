import { useState } from 'react';
import { useProjectStore } from '../project/projectStore';

const GlossaryManager = ({ onClose }: { onClose: () => void }) => {
  const project = useProjectStore((s) => s.project);
  const updateGlossary = useProjectStore((s) => s.updateGlossary);
  const glossary = project?.settings?.glossary || {};

  const [entries, setEntries] = useState<{ src: string; tgt: string }[]>(
    Object.entries(glossary).map(([src, tgt]) => ({ src, tgt }))
  );

  const handleSave = () => {
    const newGlossary: Record<string, string> = {};
    entries.forEach((e) => {
      if (e.src.trim() && e.tgt.trim()) {
        newGlossary[e.src.trim()] = e.tgt.trim();
      }
    });
    updateGlossary(newGlossary);
    onClose();
  };

  const addEntry = () => setEntries([...entries, { src: '', tgt: '' }]);
  
  const updateEntry = (index: number, field: 'src' | 'tgt', value: string) => {
    const newEntries = [...entries];
    newEntries[index][field] = value;
    setEntries(newEntries);
  };

  const removeEntry = (index: number) => {
    const newEntries = [...entries];
    newEntries.splice(index, 1);
    setEntries(newEntries);
  };

  return (
    <div className="settings-overlay" style={{ zIndex: 9999 }}>
      <div className="settings-modal" style={{ maxWidth: 600 }}>
        <div className="settings-header">
          <h2 className="settings-title">Glossaire & Mémoire</h2>
          <button type="button" className="btn-icon" onClick={onClose}>✕</button>
        </div>

        <div className="settings-body flex flex-col gap-4">
          <p className="text-sm text-gray-400">
            Définissez les termes spécifiques (noms de personnages, attaques, tutoiement/vouvoiement) qui doivent être respectés par le traducteur.
          </p>

          <div className="flex flex-col gap-2 max-h-64 overflow-y-auto">
            {entries.map((entry, idx) => (
              <div key={idx} className="flex gap-2 items-center">
                <input
                  type="text"
                  className="field-input flex-1"
                  placeholder="Terme source (ex: Goku)"
                  value={entry.src}
                  onChange={(e) => updateEntry(idx, 'src', e.target.value)}
                />
                <span className="text-gray-500">→</span>
                <input
                  type="text"
                  className="field-input flex-1"
                  placeholder="Traduction forcée"
                  value={entry.tgt}
                  onChange={(e) => updateEntry(idx, 'tgt', e.target.value)}
                />
                <button type="button" className="btn-icon text-red-500 hover:text-red-400" onClick={() => removeEntry(idx)}>✕</button>
              </div>
            ))}
          </div>

          <button type="button" className="btn btn-ghost" style={{ alignSelf: 'flex-start' }} onClick={addEntry}>
            + Ajouter un terme
          </button>

        </div>
        
        <div className="settings-footer" style={{ padding: 16, borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="btn btn-ghost" onClick={onClose}>Annuler</button>
          <button type="button" className="btn btn-primary" onClick={handleSave}>Enregistrer</button>
        </div>
      </div>
    </div>
  );
};

export default GlossaryManager;
