import { useEffect } from "react";
import { useAuthStore } from "@/store/auth";
import { useResultsStore } from "@/store/results";
import * as api from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";

export default function ProfilePage() {
  const user = useAuthStore((s) => s.user);
  const history = useResultsStore((s) => s.history);
  const setHistory = useResultsStore((s) => s.setHistory);
  const navigate = useNavigate();
  
  // Prepend API base URL to image URLs
  const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001";
  const getFullUrl = (url: string) => url.startsWith('http') ? url : `${baseURL}${url}`;

  useEffect(() => {
    api.listResults().then(async (list) => {
      const detailed = await Promise.all(list.map((s) => api.getResult(s.id)));
      setHistory(detailed);
    });
  }, [setHistory]);

  return (
    <div className="container py-10 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">Name</div>
          <div className="font-medium">{user?.name}</div>
          <div className="text-sm text-muted-foreground mt-4">Email</div>
          <div className="font-medium">{user?.email}</div>
        </CardContent>
      </Card>

      <div>
        <h2 className="text-lg font-semibold mb-3">History</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {history.map((r) => (
            <button key={r.id} onClick={() => navigate(`/results/${r.id}`)} className="text-left">
              <Card>
                <CardContent className="p-0">
                  <img src={getFullUrl(r.overlayUrl)} alt="Result thumbnail" className="w-full h-40 object-cover rounded-t-lg" />
                  <div className="p-3 text-sm flex items-center justify-between">
                    <span className="font-medium">{new Date(r.createdAt).toLocaleString()}</span>
                    <span className="text-muted-foreground">{r.id}</span>
                  </div>
                </CardContent>
              </Card>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
