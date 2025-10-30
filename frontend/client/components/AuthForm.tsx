import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuthStore } from "@/store/auth";
import { toast } from "sonner";

const signInSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
});
const signUpSchema = signInSchema.extend({ name: z.string().min(2) });

export function AuthForm({ mode, onSuccess }: { mode: "signin" | "signup"; onSuccess: () => void }) {
  const isSignUp = mode === "signup";
  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<any>({
    resolver: zodResolver(isSignUp ? signUpSchema : signInSchema),
  });
  const signIn = useAuthStore((s) => s.signIn);
  const signUp = useAuthStore((s) => s.signUp);

  const onSubmit = async (values: any) => {
    try {
      if (isSignUp) await signUp(values.name, values.email, values.password);
      else await signIn(values.email, values.password);
      toast.success("Welcome");
      onSuccess();
    } catch (e: any) {
      toast.error(e?.message || "Authentication failed");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      {isSignUp && (
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" {...register("name")} aria-invalid={!!errors.name} />
          {errors.name && <p className="text-sm text-destructive">{String(errors.name.message)}</p>}
        </div>
      )}
      <div className="space-y-2">
        <Label htmlFor="email">Email</Label>
        <Input id="email" type="email" {...register("email")} aria-invalid={!!errors.email} />
        {errors.email && <p className="text-sm text-destructive">{String(errors.email.message)}</p>}
      </div>
      <div className="space-y-2">
        <Label htmlFor="password">Password</Label>
        <Input id="password" type="password" {...register("password")} aria-invalid={!!errors.password} />
        {errors.password && <p className="text-sm text-destructive">{String(errors.password.message)}</p>}
      </div>
      <Button disabled={isSubmitting} className="w-full" type="submit">
        {isSubmitting ? "Submitting..." : isSignUp ? "Sign Up" : "Sign In"}
      </Button>
    </form>
  );
}
