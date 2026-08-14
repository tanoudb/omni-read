import { useState, useEffect } from 'react';
import axios from 'axios';

interface BatchStatus {
  status: 'idle' | 'running' | 'done' | 'error';
  total: number;
  processed: number;
  current_file: string;
  errors: Array<{ file: string; error: string }>;
}

const BatchManager = ({ onClose }: { onClose: () => void }) => {
  const [inputDir, setInputDir] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [status, setStatus] = useState<BatchStatus>({
    status: 'idle', total: 0, processed: 0, current_file: '', errors: []
  });

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (status.status === 'running') {
      interval = setInterval(async () => {
        try {
          const res = await axios.get('http://127.0.0.1:8000/batch/status');
          setStatus(res.data);
        } catch (e) {
          console.error(e);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [status.status]);

  const handleStart = async () => {
    if (!inputDir || !outputDir) return;
    try {
      await axios.post('http://127.0.0.1:8000/batch', {
        input_dir: inputDir,
        output_dir: outputDir
      });
      setStatus((s) => ({ ...s, status: 'running' }));
    } catch (e) {
      console.error(e);
      alert("Erreur de lancement du batch.");
    }
  };

  const progress = status.total > 0 ? (status.processed / status.total) * 100 : 0;

  return (
    <div className="settings-overlay" style={{ zIndex: 9999 }}>
      <div className="settings-modal" style={{ maxWidth: 600 }}>
        <div className="settings-header">
          <h2 className="settings-title">Batch Processing (Dossiers complets)</h2>
          <button type="button" className="btn-icon" onClick={onClose}>✕</button>
        </div>

        <div className="settings-body flex flex-col gap-4">
          <div className="field-group">
            <label className="field-label">Dossier source (Chemin absolu)</label>
            <input 
              type="text" 
              className="field-input" 
              placeholder="Ex: C:/Manga/Chapitre 1"
              value={inputDir}
              onChange={(e) => setInputDir(e.target.value)}
              disabled={status.status === 'running'}
            />
          </div>
          
          <div className="field-group">
            <label className="field-label">Dossier de destination (Chemin absolu)</label>
            <input 
              type="text" 
              className="field-input" 
              placeholder="Ex: C:/Manga/Chapitre 1_Traduit"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              disabled={status.status === 'running'}
            />
          </div>

          <button 
            type="button" 
            className="btn btn-primary" 
            onClick={handleStart}
            disabled={status.status === 'running' || !inputDir || !outputDir}
          >
            {status.status === 'running' ? 'Traitement en cours...' : 'Démarrer le Batch'}
          </button>

          {status.status !== 'idle' && (
            <div className="p-4 bg-gray-900 border border-gray-700 rounded-lg mt-4">
              <div className="flex justify-between text-xs text-gray-400 mb-2">
                <span>Progression</span>
                <span>{status.processed} / {status.total} fichiers</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 mb-4 overflow-hidden">
                <div className="bg-accent h-2 transition-all duration-300" style={{ width: `${progress}%` }} />
              </div>
              
              <div className="text-xs text-gray-500 mb-2 truncate" title={status.current_file}>
                Traitement actuel : {status.current_file || 'Terminé'}
              </div>

              {status.errors.length > 0 && (
                <div className="mt-4 p-2 bg-red-900/20 border border-red-500/30 rounded text-red-400 text-xs max-h-32 overflow-y-auto">
                  <div className="font-bold mb-1">Erreurs ({status.errors.length}) :</div>
                  {status.errors.map((err, i) => (
                    <div key={i} className="mb-1 truncate" title={err.error}>{err.file}: {err.error}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BatchManager;
