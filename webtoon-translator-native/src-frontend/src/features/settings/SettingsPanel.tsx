import { useState } from 'react';

type SettingsTab = 'general' | 'ocr' | 'translation' | 'rendering';

interface SettingsPanelProps {
  onClose: () => void;
}

const SettingsPanel = ({ onClose }: SettingsPanelProps) => {
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');

  return (
    <div className="settings-modal-overlay" role="dialog" aria-modal="true">
      <div className="settings-modal">
        <header className="settings-header">
          <h3>Settings</h3>
        </header>

        <div className="tabs-row">
          <button
            className={`tab-btn ${activeTab === 'general' ? 'is-active' : ''}`}
            type="button"
            onClick={() => setActiveTab('general')}
          >
            Général
          </button>
          <button
            className={`tab-btn ${activeTab === 'ocr' ? 'is-active' : ''}`}
            type="button"
            onClick={() => setActiveTab('ocr')}
          >
            OCR
          </button>
          <button
            className={`tab-btn ${activeTab === 'translation' ? 'is-active' : ''}`}
            type="button"
            onClick={() => setActiveTab('translation')}
          >
            Translation
          </button>
          <button
            className={`tab-btn ${activeTab === 'rendering' ? 'is-active' : ''}`}
            type="button"
            onClick={() => setActiveTab('rendering')}
          >
            Rendering
          </button>
        </div>

        <section className="settings-content">
          {activeTab === 'general' ? <p>Paramètres généraux (placeholder)</p> : null}
          {activeTab === 'ocr' ? <p>Paramètres OCR (placeholder)</p> : null}
          {activeTab === 'translation' ? <p>Paramètres Translation (placeholder)</p> : null}
          {activeTab === 'rendering' ? <p>Paramètres Rendering (placeholder)</p> : null}
        </section>

        <footer className="settings-footer">
          <button type="button" className="action-btn" onClick={onClose}>
            Save + Close
          </button>
        </footer>
      </div>
    </div>
  );
};

export default SettingsPanel;
