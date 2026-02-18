import { apiClient } from '../../shared/api/client';
import { ENDPOINTS } from '../../shared/api/endpoints';
import type { AutoJobCreateRequest, AutoJobCreateResponse, AutoJobStatusResponse } from '../../shared/types';

export const createAutoJob = async (payload: AutoJobCreateRequest): Promise<AutoJobCreateResponse> => {
  const response = await apiClient.post<AutoJobCreateResponse>(ENDPOINTS.jobsAuto, payload);
  return response.data;
};

export const fetchJobStatus = async (jobId: string, offset: number): Promise<AutoJobStatusResponse> => {
  const response = await apiClient.get<AutoJobStatusResponse>(ENDPOINTS.jobStatus(jobId), {
    params: { offset },
  });
  return response.data;
};

export const pollJobStatus = async (
  jobId: string,
  onTick: (payload: AutoJobStatusResponse) => void,
  intervalMs = 500
): Promise<AutoJobStatusResponse> => {
  let offset = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const snapshot = await fetchJobStatus(jobId, offset);
    offset = snapshot.next_offset ?? offset;
    onTick(snapshot);

    if (snapshot.status === 'done' || snapshot.status === 'failed') {
      return snapshot;
    }

    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
};
