# Brain Web Server UI Revamp Plan

**Date:** January 2026  
**Status:** Implementation Ready

## Executive Summary

Complete redesign of the ContinuonBrain web UI to match current capabilities (149+ API endpoints), eliminate wasted space, and provide accurate real-time visualizations.

## Current State Analysis

### Problems with Current UI
1. **Incomplete API Coverage** - Many endpoints not exposed in UI
2. **Wasted Space** - Large empty areas, poor information density
3. **Outdated Visuals** - Generic button-heavy interface, no real data visualization
4. **Missing Features** - No RCAN status, no safety dashboard, no swarm status
5. **Poor Mobile Experience** - Not responsive

### Current API Surface (149+ endpoints)
- **Core:** status, mode, discovery, events
- **RCAN:** discover, status, claim, release, command
- **Training:** wavecore, HOPE eval, teacher mode, chat learning
- **Safety:** hold, reset, work authorization
- **Agent:** chat, knowledge map, search, validation
- **Hardware:** camera, audio, joints, drive
- **Ownership:** pairing, transfer, status
- **Network:** WiFi, Bluetooth scanning/connecting

## Design Goals

1. **Zero Wasted Space** - Every pixel serves a purpose
2. **Complete API Coverage** - All 149+ endpoints accessible
3. **Real-Time Data** - Live updates via SSE/polling
4. **Modern Aesthetic** - Dark theme, glassmorphism, subtle animations
5. **Mobile-First** - Responsive from 320px to 4K

## New UI Architecture

### Layout: Command Center Style

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HEADER: Robot Identity | RCAN Status | Mode Switcher | Safety Status   │
├─────────────┬───────────────────────────────────────────┬───────────────┤
│             │                                           │               │
│  NAV RAIL   │           MAIN WORKSPACE                  │  AGENT RAIL   │
│  (icons)    │     (context-sensitive content)           │  (chat +      │
│             │                                           │   status)     │
│  - Dashboard│                                           │               │
│  - Control  │                                           │               │
│  - Training │                                           │               │
│  - Safety   │                                           │               │
│  - Network  │                                           │               │
│  - Settings │                                           │               │
│             │                                           │               │
├─────────────┴───────────────────────────────────────────┴───────────────┤
│ FOOTER: System Health | Memory | Uptime | Active Sessions | Build Info │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pages / Sections

#### 1. Dashboard (Home)
- **Robot Identity Card** - Name, RURI, device ID, firmware
- **Mode Status Ring** - Visual mode indicator with quick toggle
- **RCAN Sessions** - Active sessions, roles, capabilities
- **Live Metrics Grid:**
  - HOPE loop rates (fast/mid/slow Hz)
  - CMS memory utilization
  - Wave/Particle balance gauge
  - Inference latency
- **Hardware Status Cards** - Camera, Hailo NPU, OAK-D, detected accessories
- **Quick Actions** - Mode switch, safety hold, pair app

#### 2. Control Center
- **Manual Control Pad** - Joystick-style drive controls
- **Arm Control** - Joint sliders, IK targets
- **Camera Feed** - Live view with overlays
- **Teleop Actions** - Gripper, arm poses
- **Recording Toggle** - Start/stop RLDS capture

#### 3. Training Hub
- **Training Pipeline Visualization** - RLDS → JAX → TPU → Bundle flow
- **WaveCore Status** - Fast/Mid/Slow loop metrics
- **HOPE Eval Results** - Score, level, test breakdown
- **RLDS Episodes** - List, preview, export
- **Teacher Mode Panel** - Enable, pending questions, submit answers
- **Chat Learning** - Trigger sessions, view RLDS output
- **Cloud Export** - Create zip, upload status

#### 4. Safety Center
- **Ring 0 Status** - Safety kernel health
- **Protocol 66 Rules** - Active rules, categories
- **Work Authorizations** - Pending, approved, history
- **Emergency Actions** - E-stop, safety hold, reset gates
- **Anti-Subversion Status** - Threat detection, audit log

#### 5. Network & Connectivity
- **RCAN Protocol** - Identity, discovery, active claims
- **WiFi Manager** - Scan, connect, status
- **Bluetooth Manager** - Scan, pair, connected devices
- **Ownership Status** - Owner, paired apps, transfer

#### 6. Agent Intelligence
- **Chat Interface** - Full-featured chat with HOPE
- **Knowledge Map** - Visual memory graph
- **Semantic Search** - Query memories
- **Model Stack** - Active model, fallbacks
- **Personality Config** - Humor, empathy, verbosity sliders

#### 7. Settings & Admin
- **Robot Identity** - Name, owner display
- **Hardware Config** - Mock vs real, device selection
- **Training Config** - Background learner, RLDS logging
- **Advanced** - Factory reset, promote candidate model

### Component Library

#### Status Indicators
```css
.status-dot { /* Pulsing colored dot */ }
.status-ring { /* Circular progress with mode color */ }
.status-chip { /* Inline status badge */ }
```

#### Data Cards
```css
.metric-card { /* Single value with label and trend */ }
.status-card { /* Multi-field status block */ }
.action-card { /* Clickable card with icon */ }
```

#### Real-Time Elements
```css
.live-chart { /* Streaming line chart */ }
.gauge { /* Radial gauge for percentages */ }
.timeline { /* Horizontal event timeline */ }
```

### Color Palette

```css
:root {
  /* Primary */
  --bg-deep: #0a0a0f;
  --bg-surface: #12121a;
  --bg-elevated: #1a1a24;
  --bg-glass: rgba(255,255,255,0.03);
  
  /* Accent */
  --accent-primary: #6366f1;    /* Indigo */
  --accent-success: #10b981;    /* Emerald */
  --accent-warning: #f59e0b;    /* Amber */
  --accent-danger: #ef4444;     /* Red */
  --accent-info: #3b82f6;       /* Blue */
  
  /* RCAN Role Colors */
  --role-creator: #a855f7;      /* Purple */
  --role-owner: #6366f1;        /* Indigo */
  --role-leasee: #14b8a6;       /* Teal */
  --role-user: #3b82f6;         /* Blue */
  --role-guest: #64748b;        /* Slate */
  
  /* Mode Colors */
  --mode-autonomous: #10b981;
  --mode-manual: #f59e0b;
  --mode-training: #6366f1;
  --mode-idle: #64748b;
  --mode-emergency: #ef4444;
  
  /* Text */
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  
  /* Borders */
  --border-subtle: rgba(255,255,255,0.06);
  --border-default: rgba(255,255,255,0.1);
  --border-focus: var(--accent-primary);
}
```

### Typography

```css
/* Using Inter for UI, JetBrains Mono for code */
--font-ui: 'Inter', system-ui, sans-serif;
--font-mono: 'JetBrains Mono', monospace;
--font-display: 'Space Grotesk', sans-serif;
```

### Animation Tokens

```css
--ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
--duration-fast: 150ms;
--duration-normal: 250ms;
--duration-slow: 400ms;
```

## Implementation Phases

### Phase 1: Core Layout & Navigation (2 hours)
- [ ] New base template with nav rail + workspace + agent rail
- [ ] Responsive breakpoints
- [ ] Theme variables

### Phase 2: Dashboard Page (2 hours)
- [ ] Robot identity card
- [ ] Mode status ring
- [ ] Live metrics grid
- [ ] Hardware status cards
- [ ] RCAN sessions panel

### Phase 3: Control Center (1.5 hours)
- [ ] Camera feed integration
- [ ] Manual controls
- [ ] Recording toggle

### Phase 4: Training Hub (2 hours)
- [ ] Training pipeline viz
- [ ] RLDS episode list
- [ ] Teacher mode UI
- [ ] WaveCore metrics

### Phase 5: Safety Center (1 hour)
- [ ] Safety kernel status
- [ ] Work authorization UI
- [ ] Emergency actions

### Phase 6: Network & RCAN (1 hour)
- [ ] RCAN status panel
- [ ] WiFi/Bluetooth managers
- [ ] Ownership display

### Phase 7: Polish & Testing (1 hour)
- [ ] Animations
- [ ] Mobile testing
- [ ] Error states

## File Structure

```
continuonbrain/server/templates/
├── base_v2.html          # New base layout
├── dashboard.html         # Home/dashboard
├── control.html          # Control center (update)
├── training.html         # Training hub (update)
├── safety.html           # Safety center (new)
├── network.html          # Network & RCAN (new)
├── agent.html            # Agent intelligence (new)
├── settings.html         # Settings (update)
└── components/
    ├── nav_rail.html
    ├── agent_rail.html
    ├── status_cards.html
    └── metrics_grid.html

continuonbrain/server/static/
├── css/
│   ├── ui_v2.css         # New styles
│   └── components.css    # Component library
└── js/
    ├── dashboard.js      # Dashboard logic
    ├── realtime.js       # SSE/polling
    └── charts.js         # Visualization
```

## API Integration Map

| UI Section | Primary Endpoints |
|------------|-------------------|
| Dashboard | `/api/status`, `/rcan/v1/status`, `/api/loops`, `/api/runtime/hardware` |
| Control | `/api/robot/drive`, `/api/robot/joints`, `/api/camera/stream` |
| Training | `/api/training/*`, `/api/hope/*`, `/api/waves/loops` |
| Safety | `/api/safety/*`, custom work authorization endpoints |
| Network | `/rcan/v1/*`, `/api/network/*`, `/api/ownership/*` |
| Agent | `/api/chat`, `/api/agent/*`, `/api/context/*` |
| Settings | `/api/settings`, `/api/personality/*`, `/api/identity/*` |

## Success Metrics

1. **Page Load < 1s** - Initial render fast
2. **100% Endpoint Coverage** - All APIs accessible
3. **Real-time Updates < 100ms** - SSE latency
4. **Mobile Usable** - Full functionality at 375px
5. **Zero Dead Space** - Information density maximized

