import type { Project } from '../../shared/types';

export const serializeProject = (project: Project): string => JSON.stringify(project, null, 2);

export const deserializeProject = (raw: string): Project => JSON.parse(raw) as Project;
