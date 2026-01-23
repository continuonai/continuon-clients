/**
 * Shared Navigation Bar for ContinuonXR
 * Include this script to add the navigation bar to any page
 */

(function() {
    'use strict';

    const NAV_ITEMS = [
        { href: '/', icon: '🎮', text: 'Trainer', id: 'trainer' },
        { href: '/homescan', icon: '🏠', text: 'HomeScan 3D', id: 'homescan' },
        { href: '/home-explore', icon: '🗺️', text: 'Home Explore', id: 'home-explore' },
        { href: '/room-scanner', icon: '📸', text: 'Room Scanner', id: 'room-scanner' },
        { divider: true },
        { href: 'http://localhost:8083', icon: '🎲', text: '3D Game', id: 'game', external: true },
    ];

    function getCurrentPage() {
        const path = window.location.pathname;
        if (path === '/' || path === '/index.html') return 'trainer';
        if (path.includes('homescan')) return 'homescan';
        if (path.includes('home-explore')) return 'home-explore';
        if (path.includes('room-scanner')) return 'room-scanner';
        return '';
    }

    function createNavHTML() {
        const currentPage = getCurrentPage();

        const linksHTML = NAV_ITEMS.map(item => {
            if (item.divider) {
                return '<div class="continuon-nav-divider"></div>';
            }
            const isActive = item.id === currentPage;
            const target = item.external ? ' target="_blank"' : '';
            return `
                <a href="${item.href}" class="continuon-nav-link${isActive ? ' active' : ''}"${target}>
                    <span class="icon">${item.icon}</span>
                    <span class="link-text">${item.text}</span>
                </a>
            `;
        }).join('');

        return `
            <nav class="continuon-nav">
                <div class="continuon-nav-inner">
                    <a href="/" class="continuon-brand">
                        <div class="continuon-logo">🤖</div>
                        <div class="continuon-title">Continuon<span>XR</span></div>
                    </a>
                    <div class="continuon-nav-links">
                        ${linksHTML}
                        <div class="continuon-status">
                            <div class="continuon-status-dot" id="nav-status-dot"></div>
                            <span class="continuon-status-text" id="nav-status-text">Connecting...</span>
                        </div>
                    </div>
                </div>
            </nav>
        `;
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

        // Create nav element
        const navHTML = createNavHTML();
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = navHTML;
        const nav = tempDiv.firstElementChild;

        // Insert at the beginning of body
        document.body.insertBefore(nav, document.body.firstChild);
    }

    // Update status indicator (can be called from other scripts)
    window.updateNavStatus = function(connected, text) {
        const dot = document.getElementById('nav-status-dot');
        const textEl = document.getElementById('nav-status-text');
        if (dot) {
            dot.classList.toggle('connected', connected);
        }
        if (textEl && text) {
            textEl.textContent = text;
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectNav);
    } else {
        injectNav();
    }
})();
