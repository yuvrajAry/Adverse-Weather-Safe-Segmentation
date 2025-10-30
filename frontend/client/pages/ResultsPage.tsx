import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useResultsStore } from "@/store/results";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import * as api from "@/services/api";

export default function ResultsPage() {
  const params = useParams();
  const [loading, setLoading] = useState(false);
  const last = useResultsStore((s) => s.lastResult);
  const setLast = useResultsStore((s) => s.setLastResult);

  useEffect(() => {
    const id = params.id;
    if (!id) return;
    setLoading(true);
    api.getResult(id).then((r) => setLast(r)).finally(() => setLoading(false));
  }, [params.id, setLast]);

  if (!last) return <div className="container py-10 text-muted-foreground">No result yet. Upload images from Home.</div>;
  if (loading) return <div className="container py-10 text-muted-foreground">Loading result...</div>;

  return (
    <div className="container py-8">
      <ResultsDisplay result={last} />
    </div>
  );
}
