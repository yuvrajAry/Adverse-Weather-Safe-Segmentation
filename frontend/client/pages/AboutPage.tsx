export default function AboutPage() {
  return (
    <div className="container py-10 space-y-4">
      <h1 className="text-2xl font-semibold">About AW-SafeSeg</h1>
      <p className="text-muted-foreground max-w-2xl">
        AW-SafeSeg UI visualizes segmentation outputs for adverse weather scenarios like fog, rain, and low-light. Confidence heatmaps indicate uncertainty for safety-critical classes and are derived from model outputs.
      </p>
      <p className="text-muted-foreground max-w-2xl">
        This client is designed for easy integration: enable mock mode for local demos or configure the API base URL to connect to your backend.
      </p>
    </div>
  );
}
