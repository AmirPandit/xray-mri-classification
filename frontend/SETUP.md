# ML Predictor Frontend - Setup Guide

Professional React frontend for FastAPI-based Machine Learning prediction API.

## Features

✨ **Complete ML Prediction Interface**
- Single image predictions with confidence scores
- Batch processing for multiple images
- Prediction history with pagination
- Real-time health monitoring dashboard
- JWT-based authentication
- Optional heatmap visualizations

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development
- **TailwindCSS** for styling
- **shadcn/ui** component library
- **Axios** for API requests
- **React Router** for navigation
- **Tanstack Query** for data fetching

## Prerequisites

- Node.js 18+ and npm
- FastAPI backend running (default: http://localhost:8000)

## Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:8080`

## Environment Configuration

Create a `.env` file in the root directory (optional):

```env
VITE_API_URL=http://localhost:8000
```

If not set, the API URL defaults to `http://localhost:8000`

## Connecting to Your FastAPI Backend

The frontend expects the following FastAPI endpoints:

### Authentication
- `POST /token` - Login with username/password (form data)

### Health Check
- `GET /health` - Returns API status, model status, and version

### Predictions
- `POST /predict` - Single prediction with base64 image
- `POST /predict/batch` - Batch predictions
- `POST /predict/upload` - Upload file directly

### History
- `GET /history?limit=10&offset=0` - Get prediction history

### Example FastAPI Response Formats

**Health Check:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Prediction:**
```json
{
  "prediction": "cat",
  "confidence": 0.95,
  "heatmap": "base64_encoded_string",
  "processing_time": 0.234
}
```

**History:**
```json
{
  "predictions": [
    {
      "id": "uuid",
      "prediction": "dog",
      "confidence": 0.87,
      "timestamp": "2024-01-01T12:00:00Z",
      "image_size": {"width": 224, "height": 224}
    }
  ],
  "total": 100,
  "limit": 10,
  "offset": 0
}
```

## Project Structure

```
src/
├── components/
│   ├── ui/              # shadcn components
│   ├── Layout.tsx       # Main layout with navigation
│   └── FileUpload.tsx   # Drag-and-drop file upload
├── pages/
│   ├── Index.tsx        # Landing page
│   ├── Login.tsx        # Authentication
│   ├── Dashboard.tsx    # Health monitoring
│   ├── SinglePrediction.tsx
│   ├── BatchPrediction.tsx
│   └── History.tsx
├── lib/
│   └── api.ts          # API service with axios
├── types/
│   └── api.ts          # TypeScript interfaces
└── hooks/
    └── use-toast.ts    # Toast notifications
```

## Available Pages

1. **/** - Landing page with features
2. **/login** - JWT authentication
3. **/dashboard** - API health monitoring
4. **/predict** - Single image prediction
5. **/batch** - Batch image processing
6. **/history** - Prediction history with pagination

## Design System

The app uses a professional ML/AI aesthetic with:
- **Primary Colors:** Deep blues and purples
- **Accent Colors:** Bright cyan for CTAs
- **Gradients:** Smooth transitions for premium feel
- **Shadows:** Layered depth for modern UI
- **Animations:** Smooth transitions throughout

## Building for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

The build output will be in the `dist/` directory.

## Deployment

Deploy the `dist/` folder to any static hosting service:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

## Customization

### Change API URL
Update `VITE_API_URL` in your `.env` file or modify the default in `src/lib/api.ts`

### Styling
All design tokens are in `src/index.css`. Modify CSS variables to match your brand:

```css
:root {
  --primary: 245 60% 55%;     /* Main brand color */
  --accent: 190 85% 50%;      /* CTA color */
  --gradient-primary: ...;    /* Hero gradients */
}
```

### Authentication
The app stores JWT tokens in localStorage. For production, consider implementing:
- Token refresh logic
- Secure cookie storage
- OAuth providers

## Troubleshooting

**CORS Errors:**
Ensure your FastAPI backend has CORS middleware configured:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Connection Refused:**
Verify your FastAPI server is running on port 8000 or update the API URL.

## Support

For issues or questions, please refer to the FastAPI backend documentation or React/Vite documentation.

---

Built with ❤️ using Lovable
