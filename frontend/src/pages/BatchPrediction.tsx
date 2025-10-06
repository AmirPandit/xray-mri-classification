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
import { BatchPredictionResponse } from "@/types/api";
import { Loader2, Layers, TrendingUp, Clock, X, ZoomIn } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// Image Modal Component
function ImageModal({
  src,
  alt,
  isOpen,
  onClose,
}: {
  src: string;
  alt: string;
  isOpen: boolean;
  onClose: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="relative max-w-4xl max-h-[90vh] mx-4">
        <button
          onClick={onClose}
          className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors z-10"
        >
          <X className="h-8 w-8" />
        </button>
        <div className="bg-white rounded-lg overflow-hidden">
          <img
            src={src}
            alt={alt}
            className="w-full h-auto max-h-[80vh] object-contain"
          />
        </div>
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-white text-sm bg-black/50 px-3 py-1 rounded-full">
          Click anywhere to close
        </div>
      </div>
    </div>
  );
}

// Zoomable Image Component
function ZoomableImage({
  src,
  alt,
  className = "h-12 w-12 rounded object-cover",
  previewClassName = "h-12 w-12",
}: {
  src: string;
  alt: string;
  className?: string;
  previewClassName?: string;
}) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <div
        className={`relative group cursor-pointer ${previewClassName}`}
        onClick={() => setIsModalOpen(true)}
      >
        <img
          src={src}
          alt={alt}
          className={`${className} transition-transform group-hover:scale-105`}
        />
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors rounded flex items-center justify-center">
          <ZoomIn className="h-4 w-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      </div>

      <ImageModal
        src={src}
        alt={alt}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </>
  );
}

export default function BatchPrediction() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [includeHeatmaps, setIncludeHeatmaps] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);
  const [selectedImage, setSelectedImage] = useState<{
    src: string;
    alt: string;
  } | null>(null);
  const { toast } = useToast();

  const handleFilesSelected = (files: File[]) => {
    setSelectedFiles(files);
    setResult(null);
  };

  const handlePredict = async () => {
    if (selectedFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please select at least one image to predict.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      // Convert all files to base64
      const base64Images = await Promise.all(
        selectedFiles.map((file) => apiService.fileToBase64(file))
      );

      const prediction = await apiService.batchPredict({
        images: base64Images,
        include_heatmaps: includeHeatmaps,
      });

      setResult(prediction);
      toast({
        title: "Batch prediction complete",
        description: `Processed ${
          prediction.predictions.length
        } images in ${prediction.total_time.toFixed(2)}s`,
      });
    } catch (error: any) {
      toast({
        title: "Prediction failed",
        description:
          error.response?.data?.detail ||
          "Failed to process images. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Create preview URLs for original images
  const imagePreviews = selectedFiles.map((file) => URL.createObjectURL(file));

  // Calculate average confidence
  const averageConfidence = result
    ? (result.predictions.reduce((acc, p) => acc + p.confidence, 0) /
        result.predictions.length) *
      100
    : 0;

  return (
    <Layout>
      <div className="space-y-8 max-w-6xl mx-auto">
        <div>
          <h1 className="text-3xl font-bold mb-2">Batch Prediction</h1>
          <p className="text-muted-foreground">
            Upload multiple images for efficient batch processing and analysis
          </p>
        </div>

        <Card className="shadow-medium">
          <CardHeader>
            <CardTitle>Upload Images</CardTitle>
            <CardDescription>
              Select multiple image files for batch prediction
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <FileUpload onFilesSelected={handleFilesSelected} multiple={true} />

            <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
              <div className="space-y-0.5">
                <Label
                  htmlFor="heatmap-toggle-batch"
                  className="text-base font-medium"
                >
                  Include Heatmaps
                </Label>
                <p className="text-sm text-muted-foreground">
                  Generate heatmaps for all images (increases processing time)
                </p>
              </div>
              <Switch
                id="heatmap-toggle-batch"
                checked={includeHeatmaps}
                onCheckedChange={setIncludeHeatmaps}
              />
            </div>

            <Button
              variant="gradient"
              size="lg"
              className="w-full"
              onClick={handlePredict}
              disabled={selectedFiles.length === 0 || isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Processing {selectedFiles.length} images...
                </>
              ) : (
                <>
                  <Layers className="h-5 w-5" />
                  Predict Batch ({selectedFiles.length} images)
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
                Batch Results
              </CardTitle>
              <CardDescription>
                Processed {result.predictions.length} images
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-6 rounded-lg bg-card border-2">
                  <div className="text-sm opacity-90 mb-1">Total Images</div>
                  <div className="text-3xl font-bold">
                    {result.predictions.length}
                  </div>
                </div>
                <div className="p-6 rounded-lg bg-card border-2">
                  <div className="text-sm opacity-90 mb-1 flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    Processing Time
                  </div>
                  <div className="text-3xl font-bold">
                    {result.total_time.toFixed(2)}s
                  </div>
                </div>
                <div className="p-6 rounded-lg bg-card border-2 border-primary">
                  <div className="text-sm text-muted-foreground mb-1">
                    Avg. Confidence
                  </div>
                  <div className="text-3xl font-bold text-primary">
                    {averageConfidence.toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="rounded-lg border shadow-soft overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-16">#</TableHead>
                      <TableHead>Preview</TableHead>
                      <TableHead>Image</TableHead>
                      <TableHead>Prediction</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>All Predictions</TableHead>
                      {includeHeatmaps && <TableHead>Heatmap</TableHead>}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.predictions.map((prediction, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">
                          {index + 1}
                        </TableCell>
                        <TableCell>
                          <ZoomableImage
                            src={imagePreviews[index]}
                            alt={`Preview of ${selectedFiles[index]?.name}`}
                            previewClassName="h-16 w-16"
                            className="h-16 w-16 rounded-lg object-cover border-2 border-border"
                          />
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {selectedFiles[index]?.name}
                        </TableCell>
                        <TableCell className="font-semibold capitalize">
                          {prediction.prediction}
                        </TableCell>
                        <TableCell>
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="text-xs space-y-1">
                            {prediction.all_predictions &&
                              Object.entries(prediction.all_predictions).map(
                                ([key, value]) => (
                                  <div
                                    key={key}
                                    className="flex justify-between gap-2"
                                  >
                                    <span className="capitalize">{key}:</span>
                                    <span className="font-medium">
                                      {((value as number) * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                )
                              )}
                          </div>
                        </TableCell>
                        {includeHeatmaps && (
                          <TableCell>
                            {prediction.heatmap_data && (
                              <ZoomableImage
                                src={prediction.heatmap_data}
                                alt="Heatmap"
                                previewClassName="h-16 w-16"
                                className="h-16 w-16 rounded-lg object-cover border-2 border-border"
                              />
                            )}
                          </TableCell>
                        )}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Image Preview Section */}
              <div className="mt-8">
                <h3 className="text-lg font-semibold mb-4">Image Previews</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="text-center">
                      <ZoomableImage
                        src={imagePreviews[index]}
                        alt={`Preview of ${file.name}`}
                        previewClassName="h-24 w-24 mx-auto"
                        className="h-24 w-24 rounded-lg object-cover border-2 border-border mx-auto"
                      />
                      <p className="text-xs text-muted-foreground mt-2 truncate">
                        {file.name}
                      </p>
                      <p className="text-xs font-medium capitalize">
                        {result.predictions[index]?.prediction}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Global Image Modal */}
        <ImageModal
          src={selectedImage?.src || ""}
          alt={selectedImage?.alt || ""}
          isOpen={!!selectedImage}
          onClose={() => setSelectedImage(null)}
        />
      </div>
    </Layout>
  );
}
