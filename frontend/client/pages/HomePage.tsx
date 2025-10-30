import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileUpload } from "@/components/FileUpload";

export default function HomePage() {
  return (
    <div className="container py-10 space-y-6">
      <div className="grid md:grid-cols-2 gap-6 items-stretch">
        <Card>
          <CardHeader>
            <CardTitle>Adverse Weather Safe Segmentation</CardTitle>
            <CardDescription>
              Upload paired RGB + NIR images to generate a segmentation mask, confidence heatmap, and overlay. Results can be downloaded or saved to your profile.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
              <li>Secure: token-based authentication</li>
              <li>Fast: client-only with mock mode for local testing</li>
              <li>Focused: safety-critical insights via confidence heatmaps</li>
            </ul>
          </CardContent>
        </Card>
        <FileUpload />
      </div>
    </div>
  );
}
