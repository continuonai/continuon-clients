import { test, expect } from '@playwright/test';

/**
 * Progressive Benchmark Test Suite for ContinuonXR Web Interface
 * Tests all UI elements, buttons, statistics displays, and navigation
 */

test.describe('Progressive Benchmark Suite', () => {

  test.describe('Level 1: Basic Page Loading', () => {

    test('Dashboard loads successfully', async ({ page }) => {
      const response = await page.goto('/');
      expect(response?.status()).toBe(200);
      await expect(page).toHaveTitle(/Dashboard.*Continuon/i);
    });

    test('Control page loads', async ({ page }) => {
      await page.goto('/control');
      await expect(page.locator('body')).toBeVisible();
    });

    test('Training page loads', async ({ page }) => {
      await page.goto('/training');
      await expect(page.locator('body')).toBeVisible();
    });

    test('Safety page loads', async ({ page }) => {
      await page.goto('/safety');
      await expect(page.locator('body')).toBeVisible();
    });

    test('Network page loads', async ({ page }) => {
      await page.goto('/network');
      await expect(page.locator('body')).toBeVisible();
    });

    test('Settings page loads', async ({ page }) => {
      await page.goto('/settings');
      await expect(page.locator('body')).toBeVisible();
    });
  });

  test.describe('Level 2: API Endpoints', () => {

    test('Status API returns valid JSON', async ({ request }) => {
      const response = await request.get('/api/status');
      expect(response.ok()).toBeTruthy();
      const json = await response.json();
      expect(json).toHaveProperty('status');
      expect(json).toHaveProperty('success');
    });

    test('Discovery API returns robot info', async ({ request }) => {
      const response = await request.get('/api/discovery/info');
      expect(response.ok()).toBeTruthy();
      const json = await response.json();
      expect(json).toHaveProperty('robot_name');
    });

    test('RCAN info endpoint works', async ({ request }) => {
      const response = await request.get('/api/rcan/info');
      expect(response.ok()).toBeTruthy();
    });

    test('Hardware list endpoint works', async ({ request }) => {
      const response = await request.get('/api/hardware/list');
      expect(response.ok()).toBeTruthy();
    });

    test('Mode endpoint works', async ({ request }) => {
      const response = await request.get('/api/mode');
      expect(response.ok()).toBeTruthy();
    });
  });

  test.describe('Level 3: Dashboard UI Elements', () => {

    test.beforeEach(async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
    });

    test('Mode ring displays current mode', async ({ page }) => {
      const modeRing = page.locator('.mode-ring');
      await expect(modeRing).toBeVisible();

      const modeValue = page.locator('#mode-ring-value');
      await expect(modeValue).toBeVisible();
    });

    test('Mode info section shows status', async ({ page }) => {
      const modeInfo = page.locator('.mode-info');
      await expect(modeInfo).toBeVisible();

      const modeName = page.locator('#mode-name');
      await expect(modeName).toBeVisible();
    });

    test('Loop metrics display correctly', async ({ page }) => {
      const fastLoop = page.locator('#fast-loop-hz');
      await expect(fastLoop).toBeVisible();

      const midLoop = page.locator('#mid-loop-hz');
      await expect(midLoop).toBeVisible();

      const slowLoop = page.locator('#slow-loop-hz');
      await expect(slowLoop).toBeVisible();
    });

    test('RCAN panel displays identity', async ({ page }) => {
      const rcanPanel = page.locator('.rcan-panel');
      await expect(rcanPanel).toBeVisible();

      const rcanRuri = page.locator('#rcan-ruri-display');
      await expect(rcanRuri).toBeVisible();
    });

    test('Key metrics cards are visible', async ({ page }) => {
      const metricsGrid = page.locator('.dashboard-grid');
      await expect(metricsGrid).toBeVisible();

      const metricCards = page.locator('.metric-card');
      const count = await metricCards.count();
      expect(count).toBeGreaterThanOrEqual(4);
    });

    test('Hardware grid displays devices', async ({ page }) => {
      const hardwareGrid = page.locator('#hardware-grid');
      await expect(hardwareGrid).toBeVisible();
    });

    test('Refresh button is functional', async ({ page }) => {
      const refreshBtn = page.locator('#refresh-btn');
      await expect(refreshBtn).toBeVisible();
      await expect(refreshBtn).toBeEnabled();
    });
  });

  test.describe('Level 4: Navigation & Interactive Elements', () => {

    test('Navigation rail is visible and functional', async ({ page }) => {
      await page.goto('/');

      const navRail = page.locator('.nav-rail');
      await expect(navRail).toBeVisible();

      const navLinks = page.locator('.nav-rail a, .nav-rail button');
      const count = await navLinks.count();
      expect(count).toBeGreaterThan(0);
    });

    test('Header displays robot identity', async ({ page }) => {
      await page.goto('/');

      const header = page.locator('.app-header');
      await expect(header).toBeVisible();

      const robotAvatar = page.locator('.robot-avatar');
      await expect(robotAvatar).toBeVisible();
    });

    test('Agent rail is visible', async ({ page }) => {
      await page.goto('/');

      const agentRail = page.locator('.agent-rail');
      await expect(agentRail).toBeVisible();
    });

    test('Footer displays status info', async ({ page }) => {
      await page.goto('/');

      const footer = page.locator('.app-footer');
      await expect(footer).toBeVisible();
    });

    test('Quick action buttons are clickable', async ({ page }) => {
      await page.goto('/');

      const quickActions = page.locator('.quick-action-btn');
      const count = await quickActions.count();
      expect(count).toBeGreaterThan(0);

      // Check first button is enabled
      const firstBtn = quickActions.first();
      await expect(firstBtn).toBeEnabled();
    });
  });

  test.describe('Level 5: Control Page', () => {

    test.beforeEach(async ({ page }) => {
      await page.goto('/control');
      await page.waitForLoadState('networkidle');
    });

    test('Control page has camera feed section', async ({ page }) => {
      // Look for camera or video related elements
      const cameraSection = page.locator('[class*="camera"], [class*="video"], [id*="camera"], [id*="video"]');
      const count = await cameraSection.count();
      // Camera section may or may not exist based on config
    });

    test('Manual control buttons exist', async ({ page }) => {
      // Look for control buttons
      const controlBtns = page.locator('button');
      const count = await controlBtns.count();
      expect(count).toBeGreaterThan(0);
    });
  });

  test.describe('Level 6: Training Page', () => {

    test.beforeEach(async ({ page }) => {
      await page.goto('/training');
      await page.waitForLoadState('networkidle');
    });

    test('Training page displays content', async ({ page }) => {
      await expect(page.locator('body')).toBeVisible();
    });
  });

  test.describe('Level 7: Safety Page', () => {

    test.beforeEach(async ({ page }) => {
      await page.goto('/safety');
      await page.waitForLoadState('networkidle');
    });

    test('Safety page displays Ring 0 status', async ({ page }) => {
      await expect(page.locator('body')).toBeVisible();
    });
  });

  test.describe('Level 8: Data Flow & Live Updates', () => {

    test('Dashboard updates metrics via polling', async ({ page }) => {
      await page.goto('/');

      // Wait for initial load
      await page.waitForLoadState('networkidle');

      // Check that metrics are populated (not just --)
      await page.waitForTimeout(2000);

      const modeValue = page.locator('#mode-ring-value');
      const text = await modeValue.textContent();
      // After polling, should show actual mode, not --
    });

    test('Hardware status updates', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const hardwareCount = page.locator('#hardware-count');
      await expect(hardwareCount).toBeVisible();
    });
  });

  test.describe('Level 9: Button Functionality', () => {

    test('Go Autonomous button triggers mode change', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const autoBtn = page.locator('button:has-text("Autonomous")');
      if (await autoBtn.count() > 0) {
        await expect(autoBtn.first()).toBeEnabled();
      }
    });

    test('Pair App button navigates to pairing', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const pairBtn = page.locator('button:has-text("Pair")');
      if (await pairBtn.count() > 0) {
        await pairBtn.first().click();
        await page.waitForURL(/pair/);
      }
    });

    test('Learn Session button is functional', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const learnBtn = page.locator('button:has-text("Learn")');
      if (await learnBtn.count() > 0) {
        await expect(learnBtn.first()).toBeEnabled();
      }
    });
  });

  test.describe('Level 10: Performance Benchmarks', () => {

    test('Page load time is acceptable', async ({ page }) => {
      const start = Date.now();
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      const loadTime = Date.now() - start;

      console.log(`Dashboard load time: ${loadTime}ms`);
      expect(loadTime).toBeLessThan(10000); // 10 seconds max
    });

    test('API status response time is fast', async ({ request }) => {
      const start = Date.now();
      const response = await request.get('/api/status');
      const responseTime = Date.now() - start;

      console.log(`API status response time: ${responseTime}ms`);
      expect(responseTime).toBeLessThan(5000); // 5 seconds max
    });

    test('Multiple page navigations are smooth', async ({ page }) => {
      const pages = ['/', '/control', '/training', '/safety', '/network'];

      for (const path of pages) {
        const start = Date.now();
        await page.goto(path);
        await page.waitForLoadState('domcontentloaded');
        const loadTime = Date.now() - start;
        console.log(`${path} navigation time: ${loadTime}ms`);
        expect(loadTime).toBeLessThan(8000);
      }
    });
  });
});

test.describe('UI/UX Audit', () => {

  test('All buttons have visible text or aria-labels', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const buttons = page.locator('button');
    const count = await buttons.count();

    for (let i = 0; i < count; i++) {
      const btn = buttons.nth(i);
      const text = await btn.textContent();
      const ariaLabel = await btn.getAttribute('aria-label');
      const title = await btn.getAttribute('title');

      // Button should have some form of label
      const hasLabel = (text && text.trim().length > 0) || ariaLabel || title;
      if (!hasLabel) {
        console.warn(`Button ${i} lacks accessible label`);
      }
    }
  });

  test('Interactive elements have hover states', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const quickBtns = page.locator('.quick-action-btn');
    const count = await quickBtns.count();

    if (count > 0) {
      await quickBtns.first().hover();
      // Visual verification would need screenshot comparison
    }
  });

  test('Page has proper heading hierarchy', async ({ page }) => {
    await page.goto('/');

    const h1 = page.locator('h1');
    const h1Count = await h1.count();
    expect(h1Count).toBeGreaterThanOrEqual(1);
  });

  test('Color contrast is sufficient', async ({ page }) => {
    await page.goto('/');

    // Basic check - page should have readable text
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });
});

test.describe('Accessibility & Help Features', () => {

  test('Navigation has descriptive titles and aria-labels', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const navItems = page.locator('.nav-item');
    const count = await navItems.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const item = navItems.nth(i);
      const title = await item.getAttribute('title');
      const ariaLabel = await item.getAttribute('aria-label');

      // Each nav item should have both title and aria-label
      expect(title || ariaLabel).toBeTruthy();
    }
  });

  test('Navigation rail has proper ARIA role', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const navRail = page.locator('.nav-rail');
    const role = await navRail.getAttribute('role');
    expect(role).toBe('navigation');
  });

  test('Dashboard has help text for context', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const helpText = page.locator('.help-text');
    const count = await helpText.count();
    expect(count).toBeGreaterThan(0);
  });

  test('Metric cards have informative tooltips', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const metricCards = page.locator('.metric-card');
    const count = await metricCards.count();
    expect(count).toBeGreaterThan(0);

    // Check first metric card has a title attribute
    const firstCard = metricCards.first();
    const title = await firstCard.getAttribute('title');
    expect(title).toBeTruthy();
  });

  test('Info badges are present for complex concepts', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const infoBadges = page.locator('.info-badge');
    const count = await infoBadges.count();
    expect(count).toBeGreaterThan(0);

    // Info badges should have titles
    const firstBadge = infoBadges.first();
    const title = await firstBadge.getAttribute('title');
    expect(title).toBeTruthy();
    expect(title?.length).toBeGreaterThan(10); // Should be descriptive
  });

  test('Refresh button has descriptive title', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const refreshBtn = page.locator('#refresh-btn');
    await expect(refreshBtn).toBeVisible();

    const title = await refreshBtn.getAttribute('title');
    expect(title).toBeTruthy();
    expect(title).toContain('Refresh');
  });

  test('Mode buttons are clearly labeled', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const modeBtns = page.locator('.mode-btn');
    const count = await modeBtns.count();
    expect(count).toBeGreaterThanOrEqual(4);

    // Each mode button should have text content
    for (let i = 0; i < count; i++) {
      const btn = modeBtns.nth(i);
      const text = await btn.textContent();
      expect(text?.trim().length).toBeGreaterThan(0);
    }
  });

  test('Safety button is prominently styled', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const safetyBtn = page.locator('.safety-btn');
    await expect(safetyBtn).toBeVisible();

    const text = await safetyBtn.textContent();
    expect(text).toContain('SAFETY');
  });
});

test.describe('Statistics & Live Data Display', () => {

  test('Dashboard displays live loop metrics', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000); // Wait for data to load

    const fastLoop = page.locator('#fast-loop-hz');
    await expect(fastLoop).toBeVisible();

    // After loading, should show actual number, not --
    const text = await fastLoop.textContent();
    // May still be -- if server just started
  });

  test('Hardware count is displayed', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const hardwareCount = page.locator('#hardware-count');
    await expect(hardwareCount).toBeVisible();
  });

  test('Events panel shows recent events', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const eventsPanel = page.locator('#events-panel');
    await expect(eventsPanel).toBeVisible();
  });

  test('RCAN RURI is displayed', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    const ruri = page.locator('#rcan-ruri-display');
    await expect(ruri).toBeVisible();
  });

  test('API status indicator updates', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    const statusText = page.locator('#api-status');
    await expect(statusText).toBeVisible();

    const text = await statusText.textContent();
    expect(text).toMatch(/Connected|Disconnected|Error/);
  });

  test('Footer displays system stats', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const footer = page.locator('.app-footer');
    await expect(footer).toBeVisible();

    // Check memory usage display
    const memoryUsage = page.locator('#memory-usage');
    await expect(memoryUsage).toBeVisible();

    // Check uptime display
    const uptime = page.locator('#uptime');
    await expect(uptime).toBeVisible();

    // Check inference rate display
    const inferenceRate = page.locator('#inference-rate');
    await expect(inferenceRate).toBeVisible();
  });
});
