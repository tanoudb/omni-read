import { apiClient } from '../../shared/api/client';
import { ENDPOINTS } from '../../shared/api/endpoints';
import type {
  AutoJobCreateRequest,
  AutoJobCreateResponse,
  AutoJobStatusResponse,
  DetectRequest,
  DetectResponse,
  OcrRequest,
  OcrResponse,
  TranslateRequest,
  TranslateResponse,
} from '../../shared/types';

export const createJob = async (payload: AutoJobCreateRequest): Promise<AutoJobCreateResponse> => {
  const response = await apiClient.post<AutoJobCreateResponse>(ENDPOINTS.jobs, payload);
  return response.data;
};

export const getJobStatus = async (jobId: string, offset: number): Promise<AutoJobStatusResponse> => {
  const response = await apiClient.get<AutoJobStatusResponse>(ENDPOINTS.jobStatus(jobId), {
    params: { offset },
  });
  return response.data;
};

export const pollJob = async (
  jobId: string,
  onTick: (payload: AutoJobStatusResponse) => void,
  intervalMs = 500
): Promise<AutoJobStatusResponse> => {
  let offset = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const snapshot = await getJobStatus(jobId, offset);
    offset = snapshot.next_offset ?? offset;
    onTick(snapshot);

    if (snapshot.status === 'done' || snapshot.status === 'failed') {
      return snapshot;
    }

    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
};

export const runDetect = async (payload: DetectRequest): Promise<DetectResponse> => {
  const response = await apiClient.post<DetectResponse>(ENDPOINTS.detect, payload);
  return response.data;
};

export const runOcr = async (payload: OcrRequest): Promise<OcrResponse> => {
  const response = await apiClient.post<OcrResponse>(ENDPOINTS.ocr, payload);
  return response.data;
};

export const runTranslate = async (payload: TranslateRequest): Promise<TranslateResponse> => {
  const response = await apiClient.post<TranslateResponse>(ENDPOINTS.translate, payload);
  return response.data;
};
