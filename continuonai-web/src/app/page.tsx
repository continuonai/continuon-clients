import Link from 'next/link';
import { Button } from '@/components/ui/button';
import {
  Bot,
  Cpu,
  Cloud,
  Zap,
  Shield,
  ArrowRight,
  Github,
  Play,
} from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-xl">ContinuonAI</span>
          </Link>
          <nav className="hidden md:flex items-center gap-6">
            <Link href="#features" className="text-muted-foreground hover:text-foreground transition-colors">
              Features
            </Link>
            <Link href="#pricing" className="text-muted-foreground hover:text-foreground transition-colors">
              Pricing
            </Link>
            <Link href="/docs" className="text-muted-foreground hover:text-foreground transition-colors">
              Docs
            </Link>
          </nav>
          <div className="flex items-center gap-3">
            <Link href="/dashboard">
              <Button variant="ghost">Sign In</Button>
            </Link>
            <Link href="/dashboard">
              <Button>Get Started</Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 md:py-32">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-1.5 rounded-full text-sm font-medium mb-6">
              <Zap className="w-4 h-4" />
              Now with Pi0 Model Support
            </div>
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
              Train and Deploy
              <span className="text-primary block">Robot AI Models</span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              ContinuonAI is the complete platform for managing your robot fleet,
              training state-of-the-art models, and deploying AI-powered robotics
              solutions at scale.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/dashboard">
                <Button size="lg" className="gap-2">
                  Start Free Trial
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/docs">
                <Button size="lg" variant="outline" className="gap-2">
                  <Play className="w-4 h-4" />
                  Watch Demo
                </Button>
              </Link>
            </div>
          </div>

          {/* Hero Image/Dashboard Preview */}
          <div className="mt-16 relative">
            <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent z-10 pointer-events-none" />
            <div className="bg-muted rounded-xl border shadow-2xl overflow-hidden max-w-5xl mx-auto">
              <div className="bg-muted-foreground/10 px-4 py-2 border-b flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-xs text-muted-foreground ml-2">continuonai.com/dashboard</span>
              </div>
              <div className="p-8 bg-grid-pattern min-h-[400px] flex items-center justify-center">
                <div className="grid grid-cols-3 gap-4 max-w-3xl">
                  {/* Preview Cards */}
                  <div className="bg-card p-4 rounded-lg border shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-medium">Robot-01</span>
                      <span className="w-2 h-2 bg-green-500 rounded-full" />
                    </div>
                    <div className="text-2xl font-bold text-primary">Online</div>
                    <div className="text-xs text-muted-foreground mt-1">Battery: 87%</div>
                  </div>
                  <div className="bg-card p-4 rounded-lg border shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-medium">Training</span>
                      <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    </div>
                    <div className="text-2xl font-bold text-primary">65%</div>
                    <div className="text-xs text-muted-foreground mt-1">ETA: 2h 15m</div>
                  </div>
                  <div className="bg-card p-4 rounded-lg border shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-medium">Models</span>
                      <span className="w-2 h-2 bg-purple-500 rounded-full" />
                    </div>
                    <div className="text-2xl font-bold text-primary">12</div>
                    <div className="text-xs text-muted-foreground mt-1">3 ready to deploy</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-muted/50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Everything You Need for Robot AI
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From data collection to deployment, ContinuonAI provides all the tools
              you need to build and scale your robotics AI applications.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <FeatureCard
              icon={Bot}
              title="Fleet Management"
              description="Monitor and control your entire robot fleet from a single dashboard. Real-time status, telemetry, and remote control."
            />
            <FeatureCard
              icon={Cpu}
              title="Model Training"
              description="Train state-of-the-art models including Diffusion Policy, ACT, and Pi0 with our optimized training pipeline."
            />
            <FeatureCard
              icon={Cloud}
              title="Cloud Deployment"
              description="Deploy models to your robots with one click. Automatic versioning and rollback capabilities."
            />
            <FeatureCard
              icon={Zap}
              title="Fast Inference"
              description="Optimized inference engine with Hailo AI accelerator support for real-time performance on edge devices."
            />
            <FeatureCard
              icon={Shield}
              title="Secure & Private"
              description="Enterprise-grade security with end-to-end encryption. Your data stays private and secure."
            />
            <FeatureCard
              icon={Github}
              title="Open Source"
              description="Built on open standards with full API access. Integrate with your existing tools and workflows."
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="bg-primary rounded-2xl p-8 md:p-16 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-primary-foreground mb-4">
              Ready to Get Started?
            </h2>
            <p className="text-primary-foreground/80 mb-8 max-w-xl mx-auto">
              Join hundreds of robotics teams already using ContinuonAI to build
              the next generation of intelligent robots.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/dashboard">
                <Button size="lg" variant="secondary" className="gap-2">
                  Start Free Trial
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Link href="/docs">
                <Button size="lg" variant="outline" className="bg-transparent border-primary-foreground text-primary-foreground hover:bg-primary-foreground/10">
                  View Documentation
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-12">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Bot className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-bold">ContinuonAI</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <Link href="/privacy" className="hover:text-foreground transition-colors">
                Privacy
              </Link>
              <Link href="/terms" className="hover:text-foreground transition-colors">
                Terms
              </Link>
              <Link href="/docs" className="hover:text-foreground transition-colors">
                Documentation
              </Link>
              <Link href="https://github.com/continuonai" className="hover:text-foreground transition-colors">
                GitHub
              </Link>
            </div>
            <p className="text-sm text-muted-foreground">
              Copyright 2024 ContinuonAI. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-card p-6 rounded-xl border hover:shadow-lg transition-shadow">
      <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-primary" />
      </div>
      <h3 className="font-semibold text-lg mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}
