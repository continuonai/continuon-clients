const CACHE_NAME = 'continuonbrain-mobile-shell-v4';
const OFFLINE_ASSETS = [
  '/',
  '/ui',
  '/research',
  '/static/ui.css',
  '/static/client.js',
  '/static/mobile-shell.js',
  '/static/brain-viz.js',
  '/static/vendor/three/three.module.js',
  '/static/vendor/three/OrbitControls.js',
  '/static/manifest.webmanifest',
  '/static/icons/brain-icon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      try {
        const cache = await caches.open(CACHE_NAME);
        // First cache local assets (same-origin).
        await cache.addAll(OFFLINE_ASSETS);
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
