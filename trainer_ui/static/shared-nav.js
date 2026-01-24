/**
 * ContinuonXR Navigation System
 * Neural Interface / Cybernetic HUD Design
 */

(function() {
    'use strict';

    // SVG Icons - Custom designed for ContinuonXR
    const ICONS = {
        trainer: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="11" width="18" height="10" rx="2"/>
            <circle cx="8.5" cy="16" r="1.5"/>
            <circle cx="15.5" cy="16" r="1.5"/>
            <path d="M8 11V7a4 4 0 0 1 8 0v4"/>
            <path d="M12 3v2"/>
            <path d="M9 5h6"/>
        </svg>`,
        homescan: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            <path d="M9 22V12h6v10"/>
            <path d="M2 12h2M20 12h2"/>
            <path d="M12 2v2"/>
        </svg>`,
        explore: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
            <circle cx="12" cy="12" r="1"/>
        </svg>`,
        scanner: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 7V5a2 2 0 0 1 2-2h2"/>
            <path d="M17 3h2a2 2 0 0 1 2 2v2"/>
            <path d="M21 17v2a2 2 0 0 1-2 2h-2"/>
            <path d="M7 21H5a2 2 0 0 1-2-2v-2"/>
            <rect x="7" y="7" width="10" height="10" rx="1"/>
            <path d="M12 7v10"/>
            <path d="M7 12h10"/>
        </svg>`,
        house: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12L12 2l10 10"/>
            <path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"/>
            <path d="M12 21V12"/>
            <path d="M8 12h8"/>
            <circle cx="12" cy="16" r="1"/>
        </svg>`,
        grid: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
            <circle cx="6.5" cy="6.5" r="1"/>
            <circle cx="17.5" cy="17.5" r="1"/>
        </svg>`,
        logo: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="8" r="5"/>
            <path d="M12 13v8"/>
            <path d="M8 17h8"/>
            <path d="M7 21h10"/>
            <circle cx="10" cy="7" r="0.5" fill="currentColor"/>
            <circle cx="14" cy="7" r="0.5" fill="currentColor"/>
            <path d="M10 9.5a2 2 0 0 0 4 0"/>
        </svg>`
    };

    const NAV_ITEMS = [
        { href: '/', icon: 'trainer', text: 'Trainer', id: 'trainer', shortcut: '1' },
        { href: '/homescan', icon: 'homescan', text: 'HomeScan', id: 'homescan', shortcut: '2' },
        { href: '/home-explore', icon: 'explore', text: 'Explore', id: 'home-explore', shortcut: '3' },
        { href: '/room-scanner', icon: 'scanner', text: 'Scanner', id: 'room-scanner', shortcut: '4' },
        { href: '/house-viewer', icon: 'house', text: 'House POV', id: 'house-viewer', shortcut: '5' },
        { href: '/robotgrid', icon: 'grid', text: 'RobotGrid', id: 'robotgrid', shortcut: '6' },
    ];

    function getCurrentPage() {
        const path = window.location.pathname;
        if (path === '/' || path === '/index.html') return 'trainer';
        if (path.includes('homescan')) return 'homescan';
        if (path.includes('home-explore')) return 'home-explore';
        if (path.includes('room-scanner')) return 'room-scanner';
        if (path.includes('house-viewer')) return 'house-viewer';
        if (path.includes('robotgrid')) return 'robotgrid';
        return '';
    }

    function createNavHTML() {
        const currentPage = getCurrentPage();

        const linksHTML = NAV_ITEMS.map(item => {
            const isActive = item.id === currentPage;
            const icon = ICONS[item.icon] || '';
            return `
                <a href="${item.href}"
                   class="continuon-nav-link${isActive ? ' active' : ''}"
                   data-shortcut="${item.shortcut}"
                   title="${item.text} (Alt+${item.shortcut})">
                    <span class="nav-icon">${icon}</span>
                    <span class="link-text">${item.text}</span>
                </a>
            `;
        }).join('');

        return `
            <nav class="continuon-nav" role="navigation" aria-label="Main navigation">
                <div class="continuon-nav-inner">
                    <a href="/" class="continuon-brand" aria-label="ContinuonXR Home">
                        <div class="continuon-logo" aria-hidden="true">${ICONS.logo}</div>
                        <div class="continuon-title">Continuon<span>XR</span></div>
                    </a>
                    <div class="continuon-nav-links">
                        ${linksHTML}
                        <div class="continuon-nav-divider" aria-hidden="true"></div>
                        <div class="continuon-status" role="status" aria-live="polite">
                            <div class="continuon-status-dot" id="nav-status-dot" aria-hidden="true"></div>
                            <span class="continuon-status-text" id="nav-status-text">Connecting...</span>
                        </div>
                    </div>
                </div>
            </nav>
        `;
    }

    function injectStyles() {
        // Inject additional dynamic styles
        const style = document.createElement('style');
        style.id = 'continuon-nav-dynamic';
        style.textContent = `
            /* Icon styling */
            .continuon-nav-link .nav-icon svg {
                width: 18px;
                height: 18px;
            }

            .continuon-logo svg {
                width: 24px;
                height: 24px;
                color: var(--cxr-accent, #00f0ff);
            }

            /* Keyboard focus styles */
            .continuon-nav-link:focus-visible {
                outline: 2px solid var(--cxr-accent, #00f0ff);
                outline-offset: 2px;
            }

            .continuon-brand:focus-visible {
                outline: 2px solid var(--cxr-accent, #00f0ff);
                outline-offset: 4px;
            }

            /* Entry animation */
            @keyframes navSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .continuon-nav {
                animation: navSlideIn 0.4s ease-out;
            }

            .continuon-nav-link {
                animation: navSlideIn 0.4s ease-out;
                animation-fill-mode: both;
            }

            ${NAV_ITEMS.map((_, i) => `
                .continuon-nav-link:nth-child(${i + 1}) {
                    animation-delay: ${0.05 + i * 0.05}s;
                }
            `).join('')}

            /* Active link glow animation */
            @keyframes activeGlow {
                0%, 100% {
                    box-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
                }
                50% {
                    box-shadow: 0 0 16px rgba(0, 240, 255, 0.5);
                }
            }

            .continuon-nav-link.active {
                animation: activeGlow 3s ease-in-out infinite;
            }
        `;
        document.head.appendChild(style);
    }

    function injectNav() {
        // Check if nav already exists
        if (document.querySelector('.continuon-nav')) {
            return;
        }

        // Inject CSS if not already present
        if (!document.querySelector('link[href*="shared-nav.css"]')) {
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/static/shared-nav.css';
            document.head.appendChild(cssLink);
        }

        // Inject dynamic styles
        if (!document.getElementById('continuon-nav-dynamic')) {
            injectStyles();
        }

        // Create nav element
        const navHTML = createNavHTML();
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = navHTML;
        const nav = tempDiv.firstElementChild;

        // Insert at the beginning of body
        document.body.insertBefore(nav, document.body.firstChild);

        // Setup keyboard shortcuts
        setupKeyboardShortcuts();

        // Auto-connect to WebSocket if available
        autoConnectStatus();
    }

    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Alt + number for navigation
            if (e.altKey && !e.ctrlKey && !e.shiftKey) {
                const shortcut = e.key;
                const link = document.querySelector(`.continuon-nav-link[data-shortcut="${shortcut}"]`);
                if (link) {
                    e.preventDefault();
                    link.click();
                }
            }
        });
    }

    function autoConnectStatus() {
        // Try to detect WebSocket connection
        const checkConnection = () => {
            // Check if there's a global WebSocket or connection status
            if (window.wsConnected !== undefined) {
                updateNavStatus(window.wsConnected, window.wsConnected ? 'Online' : 'Offline');
            } else if (window.socket && window.socket.readyState === WebSocket.OPEN) {
                updateNavStatus(true, 'Online');
            }
        };

        // Initial check
        setTimeout(checkConnection, 1000);

        // Periodic check
        setInterval(checkConnection, 5000);
    }

    // Update status indicator (can be called from other scripts)
    window.updateNavStatus = function(connected, text) {
        const dot = document.getElementById('nav-status-dot');
        const textEl = document.getElementById('nav-status-text');

        if (dot) {
            dot.classList.toggle('connected', connected);
            dot.setAttribute('aria-label', connected ? 'Connected' : 'Disconnected');
        }

        if (textEl && text) {
            textEl.textContent = text;
        }
    };

    // Get navigation items (for external use)
    window.getNavItems = function() {
        return [...NAV_ITEMS];
    };

    // Highlight a specific nav item temporarily
    window.highlightNavItem = function(id, duration = 2000) {
        const link = document.querySelector(`.continuon-nav-link[href*="${id}"], .continuon-nav-link[class*="${id}"]`);
        if (link) {
            link.style.animation = 'none';
            link.offsetHeight; // Trigger reflow
            link.style.animation = 'activeGlow 0.5s ease-in-out 3';
            setTimeout(() => {
                link.style.animation = '';
            }, duration);
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectNav);
    } else {
        injectNav();
    }
})();
