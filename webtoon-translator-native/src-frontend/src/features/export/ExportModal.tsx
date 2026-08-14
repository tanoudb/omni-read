import { useState } from 'react';
import axios from 'axios';

const ExportModal = ({ onClose }: { onClose: () => void }) => {
  const [inputDir, setInputDir] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [format, setFormat] = useState('cbz');
  const [watermark, setWatermark] = useState('');
  const [status, setStatus] = useState<'idle' | 'exporting' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');

  const handleExport = async () => {
    if (!inputDir || !outputPath) return;
    setStatus('exporting');
    setMessage('Exportation en cours...');
    
    try {
      const res = await axios.post('http://127.0.0.1:8000/export', {
        input_dir: inputDir,
        output_path: outputPath,
        format: format,
        watermark_text: watermark,
      });
      setStatus('success');
      setMessage(res.data.message);
    } catch (e: any) {
      console.error(e);
      setStatus('error');
      setMessage(e.response?.data?.detail || "Erreur lors de l'exportation.");
    }
  };

  return (
    <div className="settings-overlay" style={{ zIndex: 9999 }}>
      <div className="settings-modal" style={{ maxWidth: 600 }}>
        <div className="settings-header">
          <h2 className="settings-title">Usine d'Exportation & Publication</h2>
          <button type="button" className="btn-icon" onClick={onClose}>✕</button>
        </div>

        <div className="settings-body flex flex-col gap-4">
          <div className="field-group">
            <label className="field-label">Dossier source des images traduites</label>
            <input 
              type="text" 
              className="field-input" 
              placeholder="Ex: C:/Manga/Chapitre 1_Traduit"
              value={inputDir}
              onChange={(e) => setInputDir(e.target.value)}
              disabled={status === 'exporting'}
            />
          </div>
          
          <div className="field-group">
            <label className="field-label">Fichier de destination</label>
            <input 
              type="text" 
              className="field-input" 
              placeholder="Ex: C:/Manga/Export/Chapitre_1.cbz"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              disabled={status === 'exporting'}
            />
          </div>

          <div className="field-row">
            <div className="field-group">
              <label className="field-label">Format</label>
              <select 
                className="style-select" 
                value={format} 
                onChange={(e) => setFormat(e.target.value)}
                disabled={status === 'exporting'}
              >
                <option value="cbz">Archive CBZ (.cbz)</option>
                <option value="zip">Archive ZIP (.zip)</option>
              </select>
            </div>
          </div>

          <div className="field-group">
            <label className="field-label">Filigrane / Watermark (sur la première page)</label>
            <input 
              type="text" 
              className="field-input" 
              placeholder="Ex: Traduit par l'équipe XYZ"
              value={watermark}
              onChange={(e) => setWatermark(e.target.value)}
              disabled={status === 'exporting'}
            />
          </div>

          <button 
            type="button" 
            className="btn btn-primary" 
            onClick={handleExport}
            disabled={status === 'exporting' || !inputDir || !outputPath}
          >
            {status === 'exporting' ? 'Exportation en cours...' : 'Lancer l\'exportation'}
          </button>

          {message && (
            <div className={`p-4 mt-2 border rounded-lg text-sm ${status === 'error' ? 'bg-red-900/20 border-red-500/30 text-red-400' : 'bg-green-900/20 border-green-500/30 text-green-400'}`}>
              {message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExportModal;
