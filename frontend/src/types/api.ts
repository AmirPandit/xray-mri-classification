export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  version: string;
}

export interface PredictionRequest {
  image: string; // base64 encoded
  include_heatmap?: boolean;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  heatmap_data?: string; // base64 encoded
  processing_time?: number;
}

export interface BatchPredictionRequest {
  images: string[]; // array of base64 encoded images
  include_heatmaps?: boolean;
}

export interface BatchPredictionResponse {
  predictions: Array<{
    prediction: string;
    confidence: number;
    all_predictions: {
      [key: string]: number;
    };
    heatmap_data?: string;
    model_version: string;
    inference_time: number;
  }>;
  total_time: number;
}

export interface HistoryResponse {
  id: string;
  prediction: string;
  confidence: number;
  timestamp: string;
  image_size: {
    width: number;
    height: number;
  };
}

// export interface HistoryResponse {
//   predictions: HistoryItem[];
//   total: number;
//   limit: number;
//   offset: number;
// }

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  full_name: string;
}
