// services/api.js
import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Backend base URL

export const startInspection = async (inspectionData) => {
  const response = await axios.post(`${API_URL}/start-inspection`, inspectionData, {
    headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
  });
  return response.data;
};

export const uploadFile = async (reportId, fileType, file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  return axios.post(`${API_URL}/upload/${reportId}/${fileType}`, formData, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem('token')}`,
      'Content-Type': 'multipart/form-data'
    }
  });
};

export const generateReport = async (reportId, format = 'pdf') => {
  return axios.post(`${API_URL}/generate-report`, {
    report_id: reportId,
    format
  }, {
    headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
  });
};

export const getReportStatus = async (reportId) => {
  return axios.get(`${API_URL}/report/${reportId}`, {
    headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
  });
};

export const downloadReport = (reportId, format) => {
  return `${API_URL}/download-report/${reportId}/${format}`;
};