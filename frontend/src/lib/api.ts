import axios, { AxiosInstance } from 'axios';
import {
  HealthCheckResponse,
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  HistoryResponse,
  AuthResponse,
  RegisterRequest,
} from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add token to requests if available
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  // Authentication
  async register(data: RegisterRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/register', data);
    // Store token
    localStorage.setItem('auth_token', response.data.access_token);
    return response.data;
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/login', {
      email,
      password,
    });
    
    // Store token
    localStorage.setItem('auth_token', response.data.access_token);
    return response.data;
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      localStorage.removeItem('auth_token');
    }
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem('auth_token');
  }

  // Health Check
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await this.client.get<HealthCheckResponse>('/health');
    return response.data;
  }

  // Single Prediction
  async predict(data: PredictionRequest): Promise<PredictionResponse> {
    const response = await this.client.post<PredictionResponse>('/predict', data);
    return response.data;
  }

  // Batch Prediction
  async batchPredict(data: BatchPredictionRequest): Promise<BatchPredictionResponse> {
    const response = await this.client.post<BatchPredictionResponse>('/predict-batch', data);
    return response.data;
  }

  // Upload Prediction
  async uploadPredict(file: File, includeHeatmap: boolean = false): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('include_heatmap', String(includeHeatmap));

    const response = await this.client.post<PredictionResponse>('/predict/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // History
  async getHistory(limit: number = 10, offset: number = 0): Promise<HistoryResponse> {
    const response = await this.client.get<HistoryResponse>('/history', {
      params: { limit, offset },
    });
    return response.data;
  }

  // Helper to convert file to base64
  async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
    });
  }
}

export const apiService = new APIService();
