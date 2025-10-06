import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Layout } from '@/components/Layout';
import { Activity, CheckCircle2, XCircle, Info, RefreshCw } from 'lucide-react';
import { apiService } from '@/lib/api';
import { HealthCheckResponse } from '@/types/api';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

export default function Dashboard() {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const { toast } = useToast();

  const fetchHealth = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.healthCheck();
      setHealth(data);
      setLastUpdated(new Date());
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to fetch health status. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Layout>
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Health Dashboard</h1>
            <p className="text-muted-foreground">
              Monitor your ML API status and performance metrics
            </p>
          </div>
          <Button variant="outline" onClick={fetchHealth} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* API Status */}
          <Card className="shadow-medium">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                API Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                {health?.status === 'healthy' ? (
                  <>
                    <CheckCircle2 className="h-8 w-8 text-green-500" />
                    <div>
                      <div className="text-2xl font-bold">Operational</div>
                      <div className="text-xs text-muted-foreground">All systems running</div>
                    </div>
                  </>
                ) : isLoading ? (
                  <div className="text-muted-foreground">Loading...</div>
                ) : (
                  <>
                    <XCircle className="h-8 w-8 text-destructive" />
                    <div>
                      <div className="text-2xl font-bold">Offline</div>
                      <div className="text-xs text-muted-foreground">Connection failed</div>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Model Status */}
          <Card className="shadow-medium">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Info className="h-4 w-4 text-muted-foreground" />
                Model Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                {health?.model_loaded ? (
                  <>
                    <CheckCircle2 className="h-8 w-8 text-green-500" />
                    <div>
                      <div className="text-2xl font-bold">Loaded</div>
                      <div className="text-xs text-muted-foreground">Ready for predictions</div>
                    </div>
                  </>
                ) : isLoading ? (
                  <div className="text-muted-foreground">Loading...</div>
                ) : (
                  <>
                    <XCircle className="h-8 w-8 text-orange-500" />
                    <div>
                      <div className="text-2xl font-bold">Not Loaded</div>
                      <div className="text-xs text-muted-foreground">Model unavailable</div>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Version Info */}
          <Card className="shadow-medium">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Info className="h-4 w-4 text-muted-foreground" />
                Version
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Info className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <div className="text-2xl font-bold">{health?.version || 'N/A'}</div>
                  <div className="text-xs text-muted-foreground">API version</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Information */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>System Information</CardTitle>
            <CardDescription>Detailed API health metrics and information</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-muted/50">
                  <div className="text-sm text-muted-foreground mb-1">Last Updated</div>
                  <div className="font-medium">{lastUpdated.toLocaleString()}</div>
                </div>
                <div className="p-4 rounded-lg bg-muted/50">
                  <div className="text-sm text-muted-foreground mb-1">Response Time</div>
                  <div className="font-medium">&lt; 100ms</div>
                </div>
                <div className="p-4 rounded-lg bg-muted/50">
                  <div className="text-sm text-muted-foreground mb-1">Uptime</div>
                  <div className="font-medium">99.9%</div>
                </div>
                <div className="p-4 rounded-lg bg-muted/50">
                  <div className="text-sm text-muted-foreground mb-1">Total Requests</div>
                  <div className="font-medium">1,234,567</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
