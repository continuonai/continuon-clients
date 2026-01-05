/**
 * realtime.js - Real-time communication client for ContinuonBrain
 *
 * Provides WebSocket connection with automatic SSE fallback.
 * Supports:
 * - Bi-directional messaging via WebSocket
 * - Channel-based subscription
 * - Automatic reconnection with exponential backoff
 * - Event dispatching to registered handlers
 */

(function (global) {
    'use strict';

    /**
     * RealtimeClient - Manages real-time connection to ContinuonBrain server
     */
    class RealtimeClient {
        constructor(options = {}) {
            this.wsUrl = options.wsUrl || this._buildWsUrl('/ws/events');
            this.sseUrl = options.sseUrl || '/api/events';
            this.autoReconnect = options.autoReconnect !== false;
            this.reconnectDelay = options.reconnectDelay || 1000;
            this.maxReconnectDelay = options.maxReconnectDelay || 30000;
            this.debug = options.debug || false;

            this._ws = null;
            this._sse = null;
            this._connectionType = null; // 'websocket' or 'sse'
            this._reconnectAttempts = 0;
            this._reconnectTimer = null;
            this._handlers = new Map();
            this._pendingRequests = new Map();
            this._requestId = 0;
            this._subscriptions = new Set(['status', 'cognitive']); // Default subscriptions
            this._connected = false;
            this._connectionId = null;
        }

        /**
         * Build WebSocket URL from current location
         */
        _buildWsUrl(path) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${protocol}//${window.location.host}${path}`;
        }

        /**
         * Start the connection (tries WebSocket first, falls back to SSE)
         */
        connect() {
            if (this._connected) {
                this._log('Already connected');
                return Promise.resolve();
            }

            return this._connectWebSocket()
                .catch(err => {
                    this._log('WebSocket failed, falling back to SSE:', err.message);
                    return this._connectSSE();
                });
        }

        /**
         * Connect via WebSocket
         */
        _connectWebSocket() {
            return new Promise((resolve, reject) => {
                try {
                    this._ws = new WebSocket(this.wsUrl);

                    this._ws.onopen = () => {
                        this._log('WebSocket connected');
                        this._connectionType = 'websocket';
                        this._connected = true;
                        this._reconnectAttempts = 0;
                        this._emit('connected', { type: 'websocket' });

                        // Send subscriptions
                        this._sendSubscriptions();
                        resolve();
                    };

                    this._ws.onclose = (event) => {
                        this._log('WebSocket closed:', event.code, event.reason);
                        this._handleDisconnect();
                    };

                    this._ws.onerror = (error) => {
                        this._log('WebSocket error:', error);
                        if (!this._connected) {
                            reject(new Error('WebSocket connection failed'));
                        }
                    };

                    this._ws.onmessage = (event) => {
                        this._handleMessage(event.data);
                    };

                    // Timeout for connection
                    setTimeout(() => {
                        if (!this._connected && this._ws.readyState !== WebSocket.OPEN) {
                            this._ws.close();
                            reject(new Error('WebSocket connection timeout'));
                        }
                    }, 5000);
                } catch (err) {
                    reject(err);
                }
            });
        }

        /**
         * Connect via SSE (fallback)
         */
        _connectSSE() {
            return new Promise((resolve, reject) => {
                try {
                    this._sse = new EventSource(this.sseUrl);

                    this._sse.onopen = () => {
                        this._log('SSE connected');
                        this._connectionType = 'sse';
                        this._connected = true;
                        this._reconnectAttempts = 0;
                        this._emit('connected', { type: 'sse' });
                        resolve();
                    };

                    this._sse.onerror = (error) => {
                        this._log('SSE error:', error);
                        if (!this._connected) {
                            reject(new Error('SSE connection failed'));
                        } else {
                            this._handleDisconnect();
                        }
                    };

                    // Listen for named events
                    const eventTypes = ['connected', 'status', 'training', 'cognitive', 'chat', 'loops', 'heartbeat', 'error'];
                    eventTypes.forEach(type => {
                        this._sse.addEventListener(type, (event) => {
                            try {
                                const data = JSON.parse(event.data);
                                this._emit(type, data);
                            } catch (err) {
                                this._log('Failed to parse SSE event:', err);
                            }
                        });
                    });

                    // Timeout for connection
                    setTimeout(() => {
                        if (!this._connected) {
                            this._sse.close();
                            reject(new Error('SSE connection timeout'));
                        }
                    }, 5000);
                } catch (err) {
                    reject(err);
                }
            });
        }

        /**
         * Handle incoming message (WebSocket only)
         */
        _handleMessage(data) {
            try {
                const message = JSON.parse(data);
                const type = message.type;

                this._log('Received:', type, message);

                // Handle command responses
                if (type === 'command_response' && message.request_id) {
                    const pending = this._pendingRequests.get(message.request_id);
                    if (pending) {
                        this._pendingRequests.delete(message.request_id);
                        if (message.success) {
                            pending.resolve(message.result);
                        } else {
                            pending.reject(new Error(message.error || 'Command failed'));
                        }
                        return;
                    }
                }

                // Handle welcome message
                if (type === 'welcome') {
                    this._connectionId = message.connection_id;
                    this._emit('welcome', message);
                    return;
                }

                // Handle events
                if (type === 'event') {
                    this._emit(message.channel, message.data);
                    return;
                }

                // Handle subscribed/unsubscribed
                if (type === 'subscribed' || type === 'unsubscribed') {
                    this._emit(type, message);
                    return;
                }

                // Handle pong
                if (type === 'pong') {
                    this._emit('pong', message);
                    return;
                }

                // Emit generic message
                this._emit(type, message);
            } catch (err) {
                this._log('Failed to parse message:', err);
            }
        }

        /**
         * Handle disconnection
         */
        _handleDisconnect() {
            const wasConnected = this._connected;
            this._connected = false;
            this._connectionType = null;

            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }
            if (this._sse) {
                this._sse.close();
                this._sse = null;
            }

            if (wasConnected) {
                this._emit('disconnected', {});
            }

            if (this.autoReconnect) {
                this._scheduleReconnect();
            }
        }

        /**
         * Schedule reconnection with exponential backoff
         */
        _scheduleReconnect() {
            if (this._reconnectTimer) {
                clearTimeout(this._reconnectTimer);
            }

            const delay = Math.min(
                this.reconnectDelay * Math.pow(2, this._reconnectAttempts),
                this.maxReconnectDelay
            );

            this._log(`Reconnecting in ${delay}ms (attempt ${this._reconnectAttempts + 1})`);

            this._reconnectTimer = setTimeout(() => {
                this._reconnectAttempts++;
                this.connect().catch(() => {
                    // Will auto-retry via _handleDisconnect
                });
            }, delay);
        }

        /**
         * Send subscriptions to server (WebSocket only)
         */
        _sendSubscriptions() {
            if (this._connectionType === 'websocket' && this._ws && this._subscriptions.size > 0) {
                this.send({
                    type: 'subscribe',
                    channels: Array.from(this._subscriptions)
                });
            }
        }

        /**
         * Subscribe to channels
         */
        subscribe(channels) {
            if (typeof channels === 'string') {
                channels = [channels];
            }
            channels.forEach(ch => this._subscriptions.add(ch));

            if (this._connected && this._connectionType === 'websocket') {
                this.send({
                    type: 'subscribe',
                    channels: channels
                });
            }
        }

        /**
         * Unsubscribe from channels
         */
        unsubscribe(channels) {
            if (typeof channels === 'string') {
                channels = [channels];
            }
            channels.forEach(ch => this._subscriptions.delete(ch));

            if (this._connected && this._connectionType === 'websocket') {
                this.send({
                    type: 'unsubscribe',
                    channels: channels
                });
            }
        }

        /**
         * Send a message (WebSocket only)
         */
        send(message) {
            if (this._connectionType !== 'websocket' || !this._ws) {
                this._log('Cannot send: not connected via WebSocket');
                return false;
            }

            try {
                this._ws.send(JSON.stringify(message));
                return true;
            } catch (err) {
                this._log('Send failed:', err);
                return false;
            }
        }

        /**
         * Send a command and wait for response (WebSocket only)
         */
        sendCommand(command, args = {}) {
            return new Promise((resolve, reject) => {
                if (this._connectionType !== 'websocket') {
                    // Fall back to HTTP API for SSE connections
                    return this._sendCommandViaHttp(command, args).then(resolve).catch(reject);
                }

                const requestId = `req_${++this._requestId}`;
                this._pendingRequests.set(requestId, { resolve, reject });

                // Set timeout for response
                setTimeout(() => {
                    if (this._pendingRequests.has(requestId)) {
                        this._pendingRequests.delete(requestId);
                        reject(new Error('Command timeout'));
                    }
                }, 30000);

                this.send({
                    type: 'command',
                    command: command,
                    args: args,
                    request_id: requestId
                });
            });
        }

        /**
         * Send command via HTTP API (for SSE fallback)
         */
        async _sendCommandViaHttp(command, args) {
            const commandMap = {
                'status': { method: 'GET', url: '/api/status' },
                'drive': { method: 'POST', url: '/api/drive', body: args },
                'mode': { method: 'POST', url: `/api/mode/${args.mode}` },
                'safety_hold': { method: 'POST', url: '/api/safety/hold' },
                'safety_reset': { method: 'POST', url: '/api/safety/reset' },
                'chat': { method: 'POST', url: '/api/chat', body: args }
            };

            const cmd = commandMap[command];
            if (!cmd) {
                throw new Error(`Unknown command: ${command}`);
            }

            const options = {
                method: cmd.method,
                headers: { 'Content-Type': 'application/json' }
            };
            if (cmd.body) {
                options.body = JSON.stringify(cmd.body);
            }

            const response = await fetch(cmd.url, options);
            return response.json();
        }

        /**
         * Send ping to server
         */
        ping() {
            if (this._connectionType === 'websocket') {
                this.send({
                    type: 'ping',
                    timestamp: Date.now()
                });
            }
        }

        /**
         * Send a streaming chat message
         *
         * @param {string} message - The message to send
         * @param {Object} options - Optional settings
         * @param {string} options.sessionId - Session ID for conversation continuity
         * @param {Array} options.history - Conversation history
         * @param {function} options.onToken - Callback for each token received
         * @param {function} options.onDone - Callback when streaming completes
         * @param {function} options.onError - Callback for errors
         * @returns {Promise} Resolves when streaming is complete
         *
         * @example
         * realtimeClient.chatStream("Hello!", {
         *     onToken: (token) => appendToChat(token),
         *     onDone: (result) => finishChat(result.full_response)
         * });
         */
        chatStream(message, options = {}) {
            return new Promise((resolve, reject) => {
                const requestId = `chat_${++this._requestId}`;
                let fullResponse = '';

                // Set up temporary handler for this request
                const handleStreamMessage = (data) => {
                    if (data.request_id !== requestId) return;

                    if (data.chunk_type === 'token') {
                        fullResponse += data.content || '';
                        if (options.onToken) {
                            options.onToken(data.content);
                        }
                    } else if (data.chunk_type === 'done') {
                        // Clean up handler
                        this.off('chat_stream', handleStreamMessage);

                        const result = {
                            full_response: data.full_response || fullResponse,
                            confidence: data.confidence,
                            model: data.model,
                            session_id: data.session_id,
                            metadata: data.metadata
                        };

                        if (options.onDone) {
                            options.onDone(result);
                        }
                        resolve(result);
                    } else if (data.chunk_type === 'error') {
                        // Clean up handler
                        this.off('chat_stream', handleStreamMessage);

                        const error = new Error(data.error || 'Chat stream error');
                        if (options.onError) {
                            options.onError(error);
                        }
                        reject(error);
                    }
                };

                // Register the handler
                this.on('chat_stream', handleStreamMessage);

                // Send the streaming chat request
                const sent = this.send({
                    type: 'chat_stream',
                    message: message,
                    history: options.history || [],
                    session_id: options.sessionId,
                    request_id: requestId
                });

                if (!sent) {
                    this.off('chat_stream', handleStreamMessage);
                    const error = new Error('Failed to send chat stream request');
                    if (options.onError) {
                        options.onError(error);
                    }
                    reject(error);
                }

                // Timeout after 60 seconds
                setTimeout(() => {
                    if (this._handlers.has('chat_stream')) {
                        const handlers = this._handlers.get('chat_stream');
                        const idx = handlers.indexOf(handleStreamMessage);
                        if (idx !== -1) {
                            handlers.splice(idx, 1);
                            const error = new Error('Chat stream timeout');
                            if (options.onError) {
                                options.onError(error);
                            }
                            reject(error);
                        }
                    }
                }, 60000);
            });
        }

        /**
         * Register event handler
         */
        on(event, handler) {
            if (!this._handlers.has(event)) {
                this._handlers.set(event, []);
            }
            this._handlers.get(event).push(handler);
            return () => this.off(event, handler);
        }

        /**
         * Remove event handler
         */
        off(event, handler) {
            if (this._handlers.has(event)) {
                const handlers = this._handlers.get(event);
                const index = handlers.indexOf(handler);
                if (index !== -1) {
                    handlers.splice(index, 1);
                }
            }
        }

        /**
         * Emit event to handlers
         */
        _emit(event, data) {
            this._log('Emit:', event, data);
            if (this._handlers.has(event)) {
                this._handlers.get(event).forEach(handler => {
                    try {
                        handler(data);
                    } catch (err) {
                        console.error('Handler error:', err);
                    }
                });
            }

            // Also emit to wildcard handlers
            if (this._handlers.has('*')) {
                this._handlers.get('*').forEach(handler => {
                    try {
                        handler(event, data);
                    } catch (err) {
                        console.error('Wildcard handler error:', err);
                    }
                });
            }
        }

        /**
         * Disconnect from server
         */
        disconnect() {
            this.autoReconnect = false;
            if (this._reconnectTimer) {
                clearTimeout(this._reconnectTimer);
                this._reconnectTimer = null;
            }
            this._handleDisconnect();
        }

        /**
         * Get connection status
         */
        get isConnected() {
            return this._connected;
        }

        /**
         * Get connection type
         */
        get connectionType() {
            return this._connectionType;
        }

        /**
         * Get connection ID (WebSocket only)
         */
        get connectionId() {
            return this._connectionId;
        }

        /**
         * Log debug message
         */
        _log(...args) {
            if (this.debug) {
                console.log('[RealtimeClient]', ...args);
            }
        }
    }

    // Create default instance
    const defaultClient = new RealtimeClient();

    // Expose to global scope
    global.RealtimeClient = RealtimeClient;
    global.realtimeClient = defaultClient;

    // Auto-connect on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            defaultClient.connect().catch(err => {
                console.warn('Real-time connection failed:', err.message);
            });
        });
    } else {
        defaultClient.connect().catch(err => {
            console.warn('Real-time connection failed:', err.message);
        });
    }

})(typeof window !== 'undefined' ? window : this);
