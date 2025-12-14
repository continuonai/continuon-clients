const CACHE_NAME = 'continuonbrain-mobile-shell-v2';
const OFFLINE_ASSETS = [
  '/',
  '/ui',
  '/research',
  '/static/ui.css',
  '/static/client.js',
  '/static/mobile-shell.js',
  '/static/brain-viz.js',
  '/static/manifest.webmanifest',
  '/static/icons/brain-icon.svg'
];

// External (CDN) modules for the 4D brain view. These will be cached best-effort.
// Offline guarantee: works after the first successful load (service worker cache warmed).
const EXTERNAL_MODULES = [
  'https://unpkg.com/three@0.160.0/build/three.module.js',
  'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      try {
        const cache = await caches.open(CACHE_NAME);
        // First cache local assets (same-origin).
        await cache.addAll(OFFLINE_ASSETS);

        // Best-effort cache external module scripts (opaque responses allowed).
        for (const url of EXTERNAL_MODULES) {
          try {
            const req = new Request(url, { mode: 'no-cors' });
            const res = await fetch(req);
            await cache.put(req, res);
          } catch (_) {
            // ignore; CDN caching is opportunistic
          }
        }
      } catch (_) {
        // ignore install cache failures; UI should still work online
      }
    })()
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') {
    return;
  }
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request).then((res) => res || caches.match('/ui') || caches.match('/')))
  );
});
