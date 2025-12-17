/**
 * ui_docs.js - Docs view logic & CSS injection
 */

(function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
        /* Teacher Mode Styles */
        .teacher-intervention-active .chat-input {
            border: 2px solid #FFD700 !important;
            background: rgba(255, 215, 0, 0.1);
        }
        .chat-message.agent_manager {
            align-self: flex-start;
            background: #2d4233;
            border-radius: 4px 12px 12px 4px;
            color: #d7ffcf;
            border-left: 3px solid #4caf50;
        }
        .chat-message.subagent {
            align-self: flex-start;
            background: #2d3855;
            border-radius: 4px 12px 12px 4px;
            color: #d0e1ff;
            border-left: 3px solid #4c89af;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.9em;
            margin-left: 20px;
        }
        .chat-message.assistant { align-self: flex-start; background: #2d3342; border-radius: 4px 12px 12px 4px; color: #cfd7ff; }
        .chat-message.user { align-self: flex-end; background: #4a5a75; border-radius: 12px 4px 4px 12px; color: #fff; }
        .chat-message.system-alert {
            background: #742a2a;
            color: #fff5f5;
            border: 1px solid #ffbcbc;
        }
        .message-bubble.success {
            background: rgba(50, 205, 50, 0.2);
            color: #fff;
            font-size: 0.9em;
        }
        #teacher-toggle.active {
            background: rgba(255, 215, 0, 0.2);
            border-color: #FFD700;
        }
        #teacher-toggle.active strong {
            color: #FFD700;
        }

        /* Docs Styles */
        .step-item { display:flex; gap:12px; margin-bottom:12px; }
        .step-number { background:#4a5568; color:#fff; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px; flex-shrink:0; }
        .step-content strong { color:#e2e8f0; display:block; margin-bottom:2px; }
        .step-content p { color:#a0aec0; margin:0; font-size:0.9em; line-height:1.4; }
    `;
    document.head.appendChild(style);
})();
