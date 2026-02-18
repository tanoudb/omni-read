import { apiClient } from '../../shared/api/client';
import { ENDPOINTS } from '../../shared/api/endpoints';
import type { RenderRequest, RenderResponse } from '../../shared/types';

export const renderPagePreview = async (payload: RenderRequest): Promise<RenderResponse> => {
  const response = await apiClient.post<RenderResponse>(ENDPOINTS.render, payload);
  return response.data;
};
