import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Layout } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { apiService } from "@/lib/api";
import { HistoryResponse } from "@/types/api";
import { History as HistoryIcon, RefreshCw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function History() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getHistory();
      setPredictions(data || []);
    } catch (error: any) {
      toast({
        title: "Error",
        description:
          error.response?.data?.detail ||
          "Failed to fetch history. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const formatDate = (dateString: string) =>
    new Date(dateString).toLocaleString();

  // Helper function to handle image_size display
  const formatImageSize = (imageSize: any) => {
    if (typeof imageSize === "string") {
      return imageSize; // Returns "unknown" as is
    }
    if (imageSize && typeof imageSize === "object") {
      return `${imageSize.width} × ${imageSize.height}`;
    }
    return "Unknown";
  };

  if (isLoading) {
    return (
      <Layout>
        <div className="text-center py-12 text-muted-foreground">
          Loading history...
        </div>
      </Layout>
    );
  }

  if (!isLoading && predictions.length === 0) {
    return (
      <Layout>
        <div className="text-center py-12">
          <HistoryIcon className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">No prediction history found</p>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Prediction History</h1>
            <p className="text-muted-foreground">
              View and analyze your past prediction records
            </p>
          </div>
          <Button variant="outline" onClick={fetchHistory} disabled={isLoading}>
            <RefreshCw
              className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
            />
          </Button>
        </div>

        <Card className="shadow-medium">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HistoryIcon className="h-5 w-5 text-primary" />
              Recent Predictions
            </CardTitle>
            <CardDescription>
              {predictions.length} total predictions recorded
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-center py-12 text-muted-foreground">
                Loading history...
              </div>
            ) : predictions.length > 0 ? (
              <div className="space-y-4">
                <div className="rounded-lg border shadow-soft overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Timestamp</TableHead>
                        <TableHead>Prediction</TableHead>
                        <TableHead>Confidence</TableHead>
                        <TableHead>Image Size</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {predictions.map((item) => (
                        <TableRow key={item.id}>
                          <TableCell className="text-muted-foreground">
                            {formatDate(item.timestamp)}
                          </TableCell>
                          <TableCell className="font-semibold">
                            {item.prediction}
                          </TableCell>
                          <TableCell>
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {formatImageSize(item.image_size)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <HistoryIcon className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">
                  No prediction history found
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Stats Card */}
        {predictions.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="shadow-soft">
              <CardContent className="pt-6">
                <div className="text-sm text-muted-foreground mb-1">
                  Average Confidence
                </div>
                <div className="text-2xl font-bold text-primary">
                  {(
                    (predictions.reduce((acc, p) => acc + p.confidence, 0) /
                      predictions.length) *
                    100
                  ).toFixed(1)}
                  %
                </div>
              </CardContent>
            </Card>
            <Card className="shadow-soft">
              <CardContent className="pt-6">
                <div className="text-sm text-muted-foreground mb-1">
                  Total Predictions
                </div>
                <div className="text-2xl font-bold text-primary">
                  {predictions.length}
                </div>
              </CardContent>
            </Card>
            <Card className="shadow-soft">
              <CardContent className="pt-6">
                <div className="text-sm text-muted-foreground mb-1">
                  Most Recent
                </div>
                <div className="text-2xl font-bold text-primary">
                  {predictions.length > 0
                    ? formatDate(predictions[0].timestamp).split(",")[0]
                    : "N/A"}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </Layout>
  );
}
