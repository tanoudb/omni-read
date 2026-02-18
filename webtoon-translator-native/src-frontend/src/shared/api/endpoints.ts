export const ENDPOINTS = {
  health: '/health',
  jobs: '/jobs',
  jobStatus: (jobId: string) => `/jobs/${jobId}`,
  detect: '/detect',
  ocr: '/ocr',
  translate: '/translate',
  remap: '/translate/remap',
  render: '/render',
  cache: '/cache',
} as const;
