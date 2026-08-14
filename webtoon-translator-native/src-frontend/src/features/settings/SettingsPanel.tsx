import { useState } from 'react';
import { useSettingsStore } from './settingsStore';

type SettingsTab = 'general' | 'ocr' | 'translation' | 'rendering';

interface SettingsPanelProps {
  onClose: () => void;
}

const SettingsPanel = ({ onClose }: SettingsPanelProps) => {
  const [tab, setTab] = useState<SettingsTab>('general');
  const apiBaseUrl = useSettingsStore((s) => s.apiBaseUrl);
  const setApiBaseUrl = useSettingsStore((s) => s.setApiBaseUrl);
  const cacheEnabled = useSettingsStore((s) => s.cacheEnabled);
  const setCacheEnabled = useSettingsStore((s) => s.setCacheEnabled);

  return (
    <div className="settings-overlay" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 className="settings-title">⚙ Paramètres</h3>
          <button type="button" className="btn-icon" onClick={onClose}>✕</button>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: 12 }}>
          {(['general', 'ocr', 'translation', 'rendering'] as SettingsTab[]).map(t => (
            <button
              key={t}
              type="button"
              className={`inspector-tab ${tab === t ? 'is-active' : ''}`}
              style={{ flex: 'none', padding: '8px 14px' }}
              onClick={() => setTab(t)}
            >
              {t === 'general' ? 'Général' : t === 'ocr' ? 'OCR' : t === 'translation' ? 'Traduction' : 'Rendu'}
            </button>
          ))}
        </div>

        <div className="settings-section">
          {tab === 'general' ? (
            <>
              <div className="settings-row">
                <span className="settings-label">URL API backend</span>
                <input
                  type="text"
                  className="settings-input"
                  value={apiBaseUrl}
                  onChange={(e) => setApiBaseUrl(e.target.value)}
                  style={{ width: 240 }}
                />
              </div>
              <div className="settings-row" style={{ marginBottom: 0 }}>
                <span className="settings-label">Cache activé</span>
                <input
                  type="checkbox"
                  checked={cacheEnabled}
                  onChange={(e) => setCacheEnabled(e.target.checked)}
                  style={{ width: 16, height: 16, cursor: 'pointer' }}
                />
              </div>
            </>
          ) : null}
          {tab === 'ocr' ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Paramètres OCR — configurable dans config/config.yaml</p>
          ) : null}
          {tab === 'translation' ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Paramètres de traduction — configurable dans config/config.yaml</p>
          ) : null}
          {tab === 'rendering' ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Paramètres de rendu — configurable dans config/config.yaml</p>
          ) : null}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button type="button" className="btn btn-primary" onClick={onClose}>Fermer</button>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;
