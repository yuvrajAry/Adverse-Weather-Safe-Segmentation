import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AuthForm } from "@/components/AuthForm";
import { useNavigate } from "react-router-dom";

import { useEffect } from "react";
import { useAuthStore } from "@/store/auth";

export default function AuthPage() {
  const navigate = useNavigate();
  const onSuccess = () => navigate("/");

  const isAuthed = useAuthStore((s) => s.isAuthed);
  useEffect(() => {
    if (isAuthed) navigate("/");
  }, [isAuthed, navigate]);

  return (
    <div className="container py-12">
      <div className="max-w-md mx-auto">
        <Card>
          <CardHeader>
            <CardTitle>Welcome to AW-SafeSeg</CardTitle>
            <CardDescription>Sign in or create an account to continue.</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="signin" className="w-full">
              <TabsList className="w-full grid grid-cols-2">
                <TabsTrigger value="signin">Sign In</TabsTrigger>
                <TabsTrigger value="signup">Sign Up</TabsTrigger>
              </TabsList>
              <TabsContent value="signin">
                <AuthForm mode="signin" onSuccess={onSuccess} />
              </TabsContent>
              <TabsContent value="signup">
                <AuthForm mode="signup" onSuccess={onSuccess} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
