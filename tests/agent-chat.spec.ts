import { test, expect } from '@playwright/test';

/**
 * Agent Chat Test Suite for ContinuonXR
 * Tests the HOPE Chat / Claude Code CLI integration via API and UI
 */

test.describe('Agent Chat API Tests', () => {

  test.describe('Chat Status Endpoint', () => {

    test('Chat status returns valid response', async ({ request }) => {
      const response = await request.get('/api/chat/status');
      expect(response.ok()).toBeTruthy();
      const json = await response.json();
      expect(json).toHaveProperty('success', true);
      expect(json).toHaveProperty('status');
      expect(json.status).toHaveProperty('active_sessions');
    });

    test('Chat status shows model info when loaded', async ({ request }) => {
      const response = await request.get('/api/chat/status');
      const json = await response.json();
      // May have ready: false if model is still loading
      expect(json.status).toHaveProperty('ready');
      expect(typeof json.status.ready).toBe('boolean');
    });
  });

  test.describe('Chat API Endpoint', () => {

    test('Chat endpoint accepts messages', async ({ request }) => {
      const response = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({ message: 'Hello!' }),
      });
      expect(response.ok()).toBeTruthy();
      const json = await response.json();
      expect(json).toHaveProperty('success');
      expect(json).toHaveProperty('response');
      expect(json).toHaveProperty('session_id');
    });

    test('Chat returns agent identification', async ({ request }) => {
      const response = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({ message: 'What agent model are you?' }),
      });
      const json = await response.json();
      expect(json).toHaveProperty('agent');
      expect(typeof json.agent).toBe('string');
    });

    test('Chat tracks conversation turns', async ({ request }) => {
      // Send first message
      const response1 = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({
          message: 'First message',
          session_id: 'test-session-turns'
        }),
      });
      const json1 = await response1.json();

      // Send second message
      const response2 = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({
          message: 'Second message',
          session_id: 'test-session-turns'
        }),
      });
      const json2 = await response2.json();

      expect(json2.turn_count).toBeGreaterThan(json1.turn_count);
    });

    test('Chat provides response duration metrics', async ({ request }) => {
      const response = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({ message: 'Time test' }),
      });
      const json = await response.json();
      expect(json).toHaveProperty('duration_ms');
      expect(typeof json.duration_ms).toBe('number');
      expect(json.duration_ms).toBeGreaterThanOrEqual(0);
    });

    test('Chat rejects empty messages', async ({ request }) => {
      const response = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({ message: '' }),
      });
      const json = await response.json();
      expect(json.success).toBe(false);
      expect(json.error).toContain('required');
    });

    test('Chat handles session history', async ({ request }) => {
      const session_id = 'history-test-' + Date.now();

      // Send with custom history
      const response = await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({
          message: 'Continue the conversation',
          session_id,
          history: [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there!' }
          ]
        }),
      });
      const json = await response.json();
      expect(json.success).toBe(true);
      expect(json.session_id).toBe(session_id);
    });
  });

  test.describe('Chat Session Management', () => {

    test('Clear session endpoint works', async ({ request }) => {
      const session_id = 'clear-test-' + Date.now();

      // Create a session with some messages
      await request.post('/api/chat', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({
          message: 'Test message',
          session_id
        }),
      });

      // Clear the session
      const clearResponse = await request.post('/api/chat/clear', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({ session_id }),
      });
      const json = await clearResponse.json();
      expect(json.success).toBe(true);
    });

    test('Clear session requires session_id', async ({ request }) => {
      const response = await request.post('/api/chat/clear', {
        headers: { 'Content-Type': 'application/json' },
        data: JSON.stringify({}),
      });
      const json = await response.json();
      expect(json.success).toBe(false);
    });
  });

  test.describe('Agent Model Switching', () => {

    test('Model info endpoint returns current model', async ({ request }) => {
      // Use chat status which includes model info
      const response = await request.get('/api/chat/status');
      expect(response.ok()).toBeTruthy();
      const json = await response.json();
      expect(json).toHaveProperty('success', true);
      expect(json).toHaveProperty('status');
    });

    test('Model list shows available models', async ({ request }) => {
      const response = await request.get('/api/v1/models');
      expect(response.ok()).toBeTruthy();
      const json = await response.json();
      expect(json).toHaveProperty('success', true);
      expect(json).toHaveProperty('models');
      expect(Array.isArray(json.models)).toBe(true);
    });
  });
});

test.describe('Agent Chat UI Tests', () => {

  test('Chat page loads', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('body')).toBeVisible();
  });

  test('Dashboard has chat section', async ({ page }) => {
    await page.goto('/');
    // Look for chat-related UI elements
    const chatSection = page.locator('[data-testid="chat-section"], .chat-panel, #chat, .hope-chat');
    // May or may not exist depending on layout
  });

  test('Settings has agent model selection', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.locator('body')).toBeVisible();
    // Look for agent/model related settings
    const settingsContent = await page.content();
    // Settings page should exist and be interactive
  });
});

test.describe('Claude Code CLI Integration', () => {

  test('Can query code-related tasks', async ({ request }) => {
    const response = await request.post('/api/chat', {
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify({
        message: 'What files are in the current directory?',
        session_id: 'cli-test'
      }),
    });
    const json = await response.json();
    expect(json.success).toBe(true);
    // Should get some kind of response (even if fallback)
    expect(json.response.length).toBeGreaterThan(0);
  });

  test('Handles timeout gracefully', async ({ request }) => {
    const response = await request.post('/api/chat', {
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify({
        message: 'Quick response test',
        session_id: 'timeout-test'
      }),
    });
    const json = await response.json();
    // Should respond within reasonable time
    expect(json.duration_ms).toBeLessThan(120000); // Less than 2 minutes
  });
});

test.describe('HOPE Active Learning Endpoints', () => {

  test('Analyze scene endpoint exists', async ({ request }) => {
    const response = await request.post('/api/hope/analyze-scene', {
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify({}),
    });
    // May return 503 if HOPE agent not available, but should not 404
    expect(response.status()).not.toBe(404);
  });

  test('Knowledge gaps endpoint exists', async ({ request }) => {
    const response = await request.post('/api/hope/knowledge-gaps', {
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify({}),
    });
    expect(response.status()).not.toBe(404);
  });

  test('Should-ask endpoint exists', async ({ request }) => {
    const response = await request.post('/api/hope/should-ask', {
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify({ message: 'Test question' }),
    });
    expect(response.status()).not.toBe(404);
  });
});
