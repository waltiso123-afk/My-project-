import axios from "axios";

const BACKEND = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND}/api`;
const ax = axios.create({ baseURL: API });

export const getSummary = () => ax.get("/dataset/summary").then((r) => r.data);
export const getEnvironment = () => ax.get("/environment").then((r) => r.data);
export const getMlStatus = () => ax.get("/ml/status").then((r) => r.data);
export const listDataset = (params) => ax.get("/dataset", { params }).then((r) => r.data);
export const getImage = (id) => ax.get(`/dataset/${id}`).then((r) => r.data);
export const getRepresentative = () => ax.get("/dataset/representative").then((r) => r.data);
export const getHouseGroups = (onlyMulti) =>
  ax.get("/housegroups", { params: { only_multi: !!onlyMulti } }).then((r) => r.data);
export const getCandidates = (minInliers = 12) =>
  ax.get("/housegroups/candidates", { params: { max_distance: minInliers } }).then((r) => r.data);
export const getPairs = () => ax.get("/housegroups/pairs").then((r) => r.data);
export const rulePair = (image_a, image_b, ruling) =>
  ax.post("/housegroups/pairs/ruling", { image_a, image_b, ruling }).then((r) => r.data);
export const getSpecChecklist = () => ax.get("/spec/checklist").then((r) => r.data);
export const mergeGroups = (imageIds) =>
  ax.post("/housegroups/merge", { image_ids: imageIds }).then((r) => r.data);
export const confirmGroup = (gid) => ax.post(`/housegroups/${gid}/confirm`).then((r) => r.data);
export const getSplit = () => ax.get("/split").then((r) => r.data);
export const getLeakage = () => ax.get("/split/leakage").then((r) => r.data);
export const getAnnotation = (id) => ax.get(`/annotations/${id}`).then((r) => r.data);
export const saveAnnotation = (id, body) => ax.put(`/annotations/${id}`, body).then((r) => r.data);
export const approveAnnotation = (id, body) =>
  ax.post(`/annotations/${id}/approve`, body).then((r) => r.data);
export const markNeedsReview = (id, reason) =>
  ax.post(`/annotations/${id}/needs_review`, { reason }).then((r) => r.data);
export const propose = (body) => ax.post("/ml/propose", body).then((r) => r.data);
export const getQaScripts = () => ax.get("/qa/scripts").then((r) => r.data);

export const rawImageUrl = (id) => `${API}/images/${id}/raw`;
export const thumbUrl = (id, w = 400) => `${API}/images/${id}/thumb?w=${w}`;
export const mergedMaskUrl = (id) => `${API}/merged/${id}/raw`;

export const PLANE_COLORS = [
  "#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6",
  "#06B6D4", "#EF4444", "#84CC16", "#F97316", "#14B8A6",
];
