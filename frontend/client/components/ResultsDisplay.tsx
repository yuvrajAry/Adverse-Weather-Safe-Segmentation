import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, Save } from "lucide-react";
import type { PredictResult } from "@/types/api";
import { downloadAllZip, downloadImage } from "@/lib/download";
import * as api from "@/services/api";
import { toast } from "sonner";

export function ResultsDisplay({ result }: { result: PredictResult }) {
  // Prepend API base URL to image URLs
  const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001";
  const getFullUrl = (url: string | undefined) => {
    if (!url) return '';  // Handle undefined/null case
    return url.startsWith('http') ? url : `${baseURL}${url}`;
  };
  
  // Debug logging
  console.log('ResultsDisplay - baseURL:', baseURL);
  console.log('ResultsDisplay - result:', result);
  console.log('ResultsDisplay - originalUrl:', result.originalUrl);
  console.log('ResultsDisplay - full URL:', getFullUrl(result.originalUrl));
  
  const items = [
    { title: "Original", url: getFullUrl(result.originalUrl) },
    { title: "Segmentation Mask", url: getFullUrl(result.maskUrl) },
    { title: "Confidence Heatmap", url: getFullUrl(result.heatmapUrl) },
    { title: "Overlay", url: getFullUrl(result.overlayUrl) },
  ];

  const onSave = async () => {
    const { ok } = await api.saveResult(result.id);
    if (ok) toast.success("Saved to profile");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">Result ID: {result.id}</div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => downloadImage(result.overlayUrl, `${result.id}-overlay.png`)}><Download className="h-4 w-4 mr-2"/> Download Overlay</Button>
          <Button onClick={() => downloadAllZip(result)}><Download className="h-4 w-4 mr-2"/> Download All (ZIP)</Button>
          <Button variant="secondary" onClick={onSave}><Save className="h-4 w-4 mr-2"/> Save to Profile</Button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {items.map((it) => (
          <Card key={it.title}>
            <CardHeader>
              <CardTitle className="text-base flex items-center justify-between">
                <span>{it.title}</span>
                <span className="text-xs text-muted-foreground">Fit to view</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative w-full overflow-hidden rounded-md bg-muted aspect-video">
                <img src={it.url} alt={it.title} className="object-contain w-full h-full" onError={(e) => { (e.currentTarget as HTMLImageElement).src = "/placeholder.svg"; }} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
