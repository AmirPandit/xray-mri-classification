import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Layout } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { FileUpload } from "@/components/FileUpload";
import { apiService } from "@/lib/api";
import { PredictionResponse } from "@/types/api";
import { Loader2, Image as ImageIcon, TrendingUp } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function SinglePrediction() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [includeHeatmap, setIncludeHeatmap] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const { toast } = useToast();

  const handleFilesSelected = (files: File[]) => {
    setSelectedFile(files[0] || null);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select an image to predict.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const prediction = await apiService.uploadPredict(
        selectedFile,
        includeHeatmap
      );
      setResult(prediction);
      toast({
        title: "Prediction complete",
        description: `Predicted: ${prediction.prediction} (${(
          prediction.confidence * 100
        ).toFixed(1)}% confidence)`,
      });
    } catch (error: any) {
      toast({
        title: "Prediction failed",
        description:
          error.response?.data?.detail ||
          "Failed to process image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <div className="space-y-8 max-w-4xl mx-auto">
        <div>
          <h1 className="text-3xl font-bold mb-2">Single Image Prediction</h1>
          <p className="text-muted-foreground">
            Upload an image to get instant ML-powered predictions with
            confidence scores
          </p>
        </div>

        <Card className="shadow-medium">
          <CardHeader>
            <CardTitle>Upload Image</CardTitle>
            <CardDescription>
              Select an image file for prediction analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <FileUpload
              onFilesSelected={handleFilesSelected}
              multiple={false}
            />

            <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
              <div className="space-y-0.5">
                <Label
                  htmlFor="heatmap-toggle"
                  className="text-base font-medium"
                >
                  Include Heatmap
                </Label>
                <p className="text-sm text-muted-foreground">
                  Generate a visual heatmap showing prediction focus areas
                </p>
              </div>
              <Switch
                id="heatmap-toggle"
                checked={includeHeatmap}
                onCheckedChange={setIncludeHeatmap}
              />
            </div>

            <Button
              variant="gradient"
              size="lg"
              className="w-full"
              onClick={handlePredict}
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <TrendingUp className="h-5 w-5" />
                  Predict
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {result && (
          <Card className="shadow-strong border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                Prediction Results
              </CardTitle>
            </CardHeader>
            <CardContent className="text-black space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-6 rounded-lg bg-gradient-primary ">
                  <div className="text-sm opacity-90 mb-1">Prediction</div>
                  <div className="text-3xl font-bold">{result.prediction}</div>
                </div>
                <div className="p-6 rounded-lg bg-gradient-accent ">
                  <div className="text-sm opacity-90 mb-1">Confidence</div>
                  <div className="text-3xl font-bold">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {result.processing_time && (
                <div className="p-4 rounded-lg bg-muted/50">
                  <div className="text-sm text-muted-foreground">
                    Processing Time
                  </div>
                  <div className="text-lg font-medium">
                    {result.processing_time.toFixed(3)}s
                  </div>
                </div>
              )}

              {result.heatmap_data && (
                <div className="space-y-2">
                  <Label className="text-base font-medium">
                    Prediction Heatmap
                  </Label>
                  <div className="rounded-lg overflow-hidden border shadow-soft">
                    <img
                      src={result.heatmap_data}
                      alt="Prediction heatmap"
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
