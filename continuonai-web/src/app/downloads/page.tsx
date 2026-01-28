import Link from 'next/link';
import { Button } from '@/components/ui/button';
import {
  Bot,
  Download,
  Smartphone,
  Monitor,
  CheckCircle,
  ArrowRight,
  Cpu,
  Mic,
  Camera,
  Gamepad2,
} from 'lucide-react';

const APK_URL = 'https://github.com/continuonai/ContinuonXR/releases/latest/download/app-release.apk';

export default function DownloadsPage() {
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
            <Link href="/#features" className="text-muted-foreground hover:text-foreground transition-colors">
              Features
            </Link>
            <Link href="/downloads" className="text-foreground font-medium">
              Downloads
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
      <section className="py-16 md:py-24 bg-gradient-to-b from-primary/5 to-background">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Download ContinuonXR
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              The mobile companion app for training and controlling your robots. 
              Powered by Nexa SDK for on-device AI.
            </p>
          </div>
        </div>
      </section>

      {/* Main Download Card */}
      <section className="py-8">
        <div className="container mx-auto px-4">
          <div className="max-w-2xl mx-auto">
            <div className="bg-card rounded-2xl border-2 border-primary/20 shadow-xl overflow-hidden">
              <div className="bg-primary/10 p-6 border-b">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 bg-primary rounded-xl flex items-center justify-center">
                    <Gamepad2 className="w-8 h-8 text-primary-foreground" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">ContinuonXR.apk</h2>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="bg-primary text-primary-foreground text-xs font-bold px-2 py-1 rounded">
                        Nexa SDK
                      </span>
                      <span className="bg-orange-500 text-white text-xs font-bold px-2 py-1 rounded">
                        NPU Optimized
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                <p className="text-muted-foreground mb-6">
                  Robot trainer app with voice commands, camera vision, and RLDS episode recording. 
                  Uses Nexa SDK for on-device vision and voice pipelines to train robots without the cloud.
                </p>
                
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-sm">Nexa Vision & Voice</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-sm">RLDS Recording</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-sm">Drive & Arm Controls</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="text-sm">WebRTC Streaming</span>
                  </div>
                </div>

                <a href={APK_URL} className="block">
                  <Button size="lg" className="w-full gap-2 text-lg py-6">
                    <Download className="w-5 h-5" />
                    Download APK
                  </Button>
                </a>

                <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                  <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400">
                    <Smartphone className="w-4 h-4" />
                    <span className="text-sm font-medium">
                      For: Samsung Galaxy XR • Android XR • Snapdragon Phones
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Installation Guide */}
      <section className="py-12">
        <div className="container mx-auto px-4">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold mb-8 text-center">How to Install</h2>
            
            <div className="space-y-6">
              <InstallStep
                number={1}
                title="Download the APK"
                description="Tap the download button above. The file will save to your Downloads folder."
                icon={Download}
              />
              <InstallStep
                number={2}
                title="Enable Unknown Sources"
                description="Go to Settings → Apps → Your Browser → Install unknown apps → Allow"
                icon={Monitor}
                note="Samsung: Settings → Biometrics and Security → Install unknown apps"
              />
              <InstallStep
                number={3}
                title="Install the App"
                description="Open Downloads, tap the APK file, and press Install."
                icon={Smartphone}
              />
              <InstallStep
                number={4}
                title="Grant Permissions"
                description="Allow microphone, camera, and storage access when prompted."
                icon={CheckCircle}
              />
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-12 bg-muted/50">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold mb-8 text-center">App Features</h2>
            
            <div className="grid md:grid-cols-3 gap-6">
              <FeatureCard
                icon={Mic}
                title="Voice Commands"
                description="Control your robot with natural voice commands processed on-device by Nexa SDK."
              />
              <FeatureCard
                icon={Camera}
                title="Camera Vision"
                description="Real-time video streaming and vision AI for object detection and scene understanding."
              />
              <FeatureCard
                icon={Gamepad2}
                title="Teleop Controls"
                description="Drive controls and 6-axis arm manipulation for teleoperation training episodes."
              />
            </div>
          </div>
        </div>
      </section>

      {/* Supported Devices */}
      <section className="py-12">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-2xl font-bold mb-4">Supported Devices</h2>
            <p className="text-muted-foreground mb-8">
              Optimized for Snapdragon devices with Hexagon NPU
            </p>
            
            <div className="flex flex-wrap justify-center gap-3">
              <DeviceChip label="Samsung Galaxy XR" primary />
              <DeviceChip label="Android XR Devices" primary />
              <DeviceChip label="Galaxy S24/S23" />
              <DeviceChip label="OnePlus 12/11" />
              <DeviceChip label="Xiaomi 14/13" />
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
              <Link href="/" className="hover:text-foreground transition-colors">
                Home
              </Link>
              <Link href="/downloads" className="hover:text-foreground transition-colors">
                Downloads
              </Link>
              <Link href="/docs" className="hover:text-foreground transition-colors">
                Documentation
              </Link>
              <Link href="https://github.com/continuonai" className="hover:text-foreground transition-colors">
                GitHub
              </Link>
            </div>
            <p className="text-sm text-muted-foreground">
              © 2024 ContinuonAI. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function InstallStep({
  number,
  title,
  description,
  icon: Icon,
  note,
}: {
  number: number;
  title: string;
  description: string;
  icon: React.ElementType;
  note?: string;
}) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold">
          {number}
        </div>
        {number < 4 && <div className="w-0.5 h-full bg-border mt-2" />}
      </div>
      <div className="flex-1 pb-6">
        <div className="flex items-center gap-2 mb-1">
          <Icon className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">{title}</h3>
        </div>
        <p className="text-muted-foreground text-sm">{description}</p>
        {note && (
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-2 bg-blue-500/10 px-3 py-2 rounded">
            {note}
          </p>
        )}
      </div>
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
    <div className="bg-card p-6 rounded-xl border">
      <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-primary" />
      </div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground text-sm">{description}</p>
    </div>
  );
}

function DeviceChip({ label, primary = false }: { label: string; primary?: boolean }) {
  return (
    <div
      className={`px-4 py-2 rounded-full text-sm font-medium ${
        primary
          ? 'bg-primary/10 text-primary border border-primary/30'
          : 'bg-muted text-muted-foreground border'
      }`}
    >
      {label}
    </div>
  );
}
