/** Milvus Admin API 客户端 —— 封装所有后端 REST 调用. */

const BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

// ── Collection ──────────────────────────────────

export interface CollectionInfo {
  name: string;
  entity_count: number;
  status: string;
  created_at: string;
  schema_fields: string[];
  index_info: Record<string, unknown>;
  backend_type: string;  // milvus_lite | milvus_server | local_numpy
}

export function listCollections(): Promise<CollectionInfo[]> {
  return request(`${BASE}/milvus/collections`);
}

export function getCollection(name: string): Promise<CollectionInfo> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}`);
}

export function deleteCollection(name: string): Promise<{ status: string }> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

// ── Partition ───────────────────────────────────

export interface PartitionInfo {
  name: string;
  row_count: number;
  created_at: string;
}

export function listPartitions(name: string): Promise<PartitionInfo[]> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/partitions`);
}

export function createPartition(name: string, partitionName: string): Promise<{ status: string }> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/partitions`, {
    method: 'POST',
    body: JSON.stringify({ name: partitionName }),
  });
}

export function deletePartition(name: string, partitionName: string): Promise<{ status: string }> {
  return request(
    `${BASE}/milvus/collections/${encodeURIComponent(name)}/partitions/${encodeURIComponent(partitionName)}`,
    { method: 'DELETE' },
  );
}

export function getCollectionStats(name: string): Promise<Record<string, number>> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/stats`);
}

// ── Index ───────────────────────────────────────

export interface IndexInfo {
  field_name: string;
  index_type: string;
  metric_type: string;
  status: string;
}

export function listIndexes(name: string): Promise<IndexInfo[]> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/indexes`);
}

export function createIndex(
  name: string,
  indexType: string,
  metricType: string,
  extraParams?: Record<string, unknown>,
): Promise<{ status: string }> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/indexes`, {
    method: 'POST',
    body: JSON.stringify({ index_type: indexType, metric_type: metricType, extra_params: extraParams }),
  });
}

export function dropIndex(name: string): Promise<{ status: string }> {
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/indexes`, { method: 'DELETE' });
}

// ── Data CRUD ───────────────────────────────────

export interface EntityRecord {
  id: number;
  prompt: string;
  optimized_prompt?: string;
  score: number;
  image_path: string;
  subject?: string;
  category?: string;
  tags: string[];
  created_at?: string;
  model_version?: string;
  // v6 VLM 内容解析字段
  semantic_text?: string;
  topic?: string;
  content_type?: string;
  diagram_type?: string;
  main_objects?: string[];
  scene_description?: string;
  style?: string;
  color_palette?: string[];
  keywords?: string[];
  knowledge_points?: string[];
  source_type?: string;
}

export function listData(
  name: string,
  params: { limit?: number; offset?: number; subject?: string; category?: string; min_score?: number },
): Promise<{ data: EntityRecord[]; total: number }> {
  const sp = new URLSearchParams();
  if (params.limit) sp.set('limit', String(params.limit));
  if (params.offset) sp.set('offset', String(params.offset));
  if (params.category) sp.set('category', params.category);
  if (params.subject) sp.set('subject', params.subject);
  if (params.min_score != null) sp.set('min_score', String(params.min_score));
  return request(`${BASE}/milvus/collections/${encodeURIComponent(name)}/data?${sp}`);
}

export function deleteEntity(name: string, id: number): Promise<{ status: string }> {
  return request(
    `${BASE}/milvus/collections/${encodeURIComponent(name)}/data/${id}`,
    { method: 'DELETE' },
  );
}

export function updateEntity(
  name: string,
  id: number,
  data: Record<string, unknown>,
): Promise<{ status: string }> {
  return request(
    `${BASE}/milvus/collections/${encodeURIComponent(name)}/data/${id}`,
    { method: 'PUT', body: JSON.stringify(data) },
  );
}

// ── Search ──────────────────────────────────────

export interface SearchResultItem {
  image_id: number;
  prompt: string;
  optimized_prompt?: string;
  score: number;
  image_path: string;
  similarity: number;
  subject?: string;
  category?: string;
  tags: string[];
}

export interface SearchResponseV2 {
  results: SearchResultItem[];
  query_time_ms: number;
  total_in_partition: number;
  query_text?: string;
  query_subject?: string;
}

export function searchByText(data: {
  text: string;
  top_k?: number;
  subject?: string;
  min_score?: number;
}): Promise<SearchResponseV2> {
  return request(`${BASE}/search/text`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function searchByImage(formData: FormData): Promise<SearchResponseV2> {
  return request(`${BASE}/search/image`, {
    method: 'POST',
    body: formData,
    headers: {}, // let browser set multipart
  });
}

export interface SearchHistoryItem {
  query: string;
  mode: string;
  subject?: string;
  result_count: number;
  timestamp: string;
}

export function getSearchHistory(limit?: number): Promise<SearchHistoryItem[]> {
  const sp = new URLSearchParams();
  if (limit) sp.set('limit', String(limit));
  return request(`${BASE}/search/history?${sp}`);
}

export interface SubjectOption {
  value: string;
  label: string;
}

export interface CategoryOption {
  value: string;
  label: string;
}

export function getSubjects(): Promise<{ subjects: SubjectOption[]; categories?: CategoryOption[] }> {
  return request(`${BASE}/search/subjects`);
}

export function getCategories(): Promise<{ categories: CategoryOption[] }> {
  return request(`${BASE}/search/categories`);
}

// ── Semantic Search (v5) ──────────────────────

export interface SemanticSearchResultItem {
  image_id: number;
  prompt: string;
  optimized_prompt?: string;
  score: number;
  image_path: string;
  subject?: string;
  category?: string;
  tags: string[];
  topic?: string;
  knowledge_points: string[];
  keywords?: string[];
  content_type?: string;
  diagram_type?: string;
  grade_level?: string;
  main_objects?: string[];
  scene_description?: string;
  style?: string;
  color_palette?: string[];
  source_type: string;
  final_score: number;
  semantic_similarity: number;
  image_similarity: number;
  tags_overlap: number;
}

export interface SemanticSearchResponse {
  results: SemanticSearchResultItem[];
  query_time_ms: number;
  total_in_partition: number;
  query_text?: string;
  query_subject?: string;
}

export function searchSemantic(data: {
  text: string;
  top_k?: number;
  subject?: string;
  category?: string;
}): Promise<SemanticSearchResponse> {
  return request(`${BASE}/search/semantic`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ── Material Upload (v5 → v6 泛化) ──────────────────────

export interface ImageContentParseResult {
  category: string;
  content_type: string;
  main_objects: string[];
  scene_description: string;
  style: string;
  color_palette: string[];
  tags: string[];
  retrieval_prompt: string;
}

// Backward-compat alias
export type EducationParseResult = ImageContentParseResult;

export interface MaterialUploadResponse {
  record_id: number;
  image_path: string;
  parse_result: ImageContentParseResult;
  semantic_text: string;
}

export function uploadMaterial(file: File): Promise<MaterialUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  return request(`${BASE}/upload_material`, {
    method: 'POST',
    body: formData,
    headers: {}, // let browser set multipart
  });
}

// ── AI Image Generation Pipeline ──────────────

export interface DimensionScore {
  dimension: string;
  score: number;
  comment: string;
}

export interface PipelineIteration {
  iteration: number;
  prompt: string;
  image_path: string;
  overall_score: number;
  dimension_scores: DimensionScore[];
  issues: string[];
  missing_elements: string[];
  suggestions: string[];
  optimized_prompt?: string;
  changes_summary?: string;
}

export interface PipelineResponse {
  final_image_path: string;
  final_image_base64?: string;
  final_prompt: string;
  final_score: number;
  total_iterations: number;
  history: PipelineIteration[];
  stopped_reason: string;
  db_record_id?: number;
  matched_prompts?: Record<string, unknown>[];
  reused_from_record_id?: number;
  stored_in_milvus?: boolean;
  stored_in_records?: boolean;
}

export interface PipelineRequest {
  prompt: string;
  model?: string;
  mode?: 'clip_enrich';
  max_iterations?: number;
  eval_threshold?: number;
  subject?: string;
  clip_top_k?: number;
  clip_min_score?: number;
  reuse_threshold?: number;
}

export function runPipeline(data: PipelineRequest): Promise<PipelineResponse> {
  return request(`/pipeline`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export interface AsyncPipelineResponse {
  task_id: string;
  image_url: string;
}

export function runPipelineAsync(data: PipelineRequest): Promise<AsyncPipelineResponse> {
  return request(`/pipeline/async`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ── Status / Debug ─────────────────────────────

export interface MilvusStatus {
  backend_type: string;
  ready: boolean;
  total_entities: number;
  by_subject: Record<string, number>;
  partitions: { name: string; row_count: number }[];
  index_info: Record<string, unknown>;
  warning: string | null;
}

export function getMilvusStatus(): Promise<MilvusStatus> {
  return request(`${BASE}/milvus/status`);
}
