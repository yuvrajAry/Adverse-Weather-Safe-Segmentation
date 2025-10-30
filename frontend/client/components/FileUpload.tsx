import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import * as api from "@/services/api";
import { useNavigate } from "react-router-dom";
import { useResultsStore } from "@/store/results";
import { Upload, Image as ImageIcon } from "lucide-react";
import { LoadingSpinner } from "@/components/LoadingSpinner";

export function FileUpload() {
  const [rgb, setRgb] = useState<File | null>(null);
  const [nir, setNir] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const setLastResult = useResultsStore((s) => s.setLastResult);
  const addToHistory = useResultsStore((s) => s.addToHistory);

  const onSubmit = async () => {
    if (!rgb || !nir) {
      toast.error("Please select both RGB and NIR images");
      return;
    }
    setLoading(true);
    try {
      const result = await api.predict({ rgb, nir });
      setLastResult(result);
      addToHistory(result);
      navigate("/results");
    } catch (e: any) {
      toast.error(e?.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const common = "flex-1 border rounded-md p-4 h-32 flex items-center justify-center text-sm text-muted-foreground hover:bg-accent transition cursor-pointer";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload Paired Images</CardTitle>
        <CardDescription>Provide RGB (PNG/JPG) and NIR (PNG/JPG). Files are sent as multipart form-data with fields rgb and nir.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className={common}>
            <input type="file" accept="image/png,image/jpeg" className="hidden" onChange={(e) => setRgb(e.target.files?.[0] || null)} />
            <div className="flex items-center gap-2"> <ImageIcon className="opacity-60"/> <span>{rgb ? rgb.name : "Select RGB image"}</span></div>
          </label>
          <label className={common}>
            <input type="file" accept="image/png,image/jpeg" className="hidden" onChange={(e) => setNir(e.target.files?.[0] || null)} />
            <div className="flex items-center gap-2"> <ImageIcon className="opacity-60"/> <span>{nir ? nir.name : "Select NIR image"}</span></div>
          </label>
        </div>
        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">Ensure both files are selected before submitting.</p>
          <Button disabled={loading} onClick={onSubmit}>
            {loading ? <LoadingSpinner label="Uploading" /> : (<span className="inline-flex items-center gap-2"><Upload className="h-4 w-4"/> Submit</span>)}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
