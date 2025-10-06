import { Navigate } from 'react-router-dom';
import { apiService } from '@/lib/api';

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = apiService.isAuthenticated();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}
