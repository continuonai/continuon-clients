# Chat Manager Agent Layout Redesign

## Overview
Redesigned the chat interface to position messages based on their source:
- **HOPE Agent**: Always on the **left side**
- **3rd Party Models** (Gemma, Phi-2, etc.): Always **centered**
- **User Input**: Always on the **right side**

## Changes Made

### 1. CSS Updates (`continuonbrain/server/static/ui.css`)

Added comprehensive styling for chat message positioning:

```css
/* HOPE Agent - Always on the left */
.chat-message.hope,
.chat-message.hope-v1,
.chat-message.agent_manager,
.chat-message.agent-manager,
.chat-message.hope_agent {
    align-self: flex-start;
    background: #2d4233;
    color: #d7ffcf;
    border-left: 3px solid #4caf50;
    border-radius: 4px 12px 12px 4px;
    margin-right: auto;
    max-width: 65%;
}

/* 3rd Party Models - Always centered */
.chat-message.subagent,
.chat-message.assistant,
.chat-message.gemma,
.chat-message.phi,
.chat-message.third_party {
    align-self: center;
    background: #2d3855;
    color: #d0e1ff;
    border-left: 3px solid #4c89af;
    border-radius: 12px;
    margin-left: auto;
    margin-right: auto;
    max-width: 60%;
}

/* User messages - Always on the right */
.chat-message.user {
    align-self: flex-end;
    background: #4a5a75;
    color: #fff;
    border-radius: 12px 4px 4px 12px;
    margin-left: auto;
    max-width: 65%;
}
```

### 2. JavaScript Updates (`continuonbrain/server/static/chat-overlay.js`)

#### Updated `appendMessage()` function
- Enhanced role detection and mapping
- Maps various role names to positioning categories:
  - HOPE roles → `hope` (left)
  - 3rd party roles → `subagent` (center)
  - User → `user` (right)
  - System → `system` (center)

#### Updated `sendChatMessage()` function
- Detects role based on `model_hint` and `delegate_model_hint`
- Sets `responseRole` appropriately:
  - `hope-v1` → `hope` (left)
  - `consult:*` or direct models → `subagent` (center)
  - Checks response metadata for agent/model info

#### Updated `onChat()` callback
- Maps incoming realtime messages to correct positioning
- Handles various role name formats

## Role Detection Logic

### HOPE Agent (Left Side)
Detected when:
- `model_hint === 'hope-v1'` or `'hope'`
- Role contains: `hope`, `agent_manager`, `agent-manager`, `hope_agent`, `hope-agent`
- Response metadata indicates HOPE agent

### 3rd Party Models (Center)
Detected when:
- `delegate_model_hint` starts with `consult:`
- `model_hint` is set and doesn't contain `hope`
- Role is: `subagent`, `assistant` (when not HOPE), `gemma`, `phi`, `third_party`
- Role contains: `gemma`, `phi`, `llm`, `model`

### User Messages (Right Side)
- Always `role === 'user'`

### System Messages (Center)
- Always `role === 'system'` or `'system-alert'`

## Visual Design

### Color Scheme
- **HOPE Agent (Left)**: Green tint (#2d4233) with green border (#4caf50)
- **3rd Party (Center)**: Blue tint (#2d3855) with blue border (#4c89af)
- **User (Right)**: Gray-blue (#4a5a75)
- **System (Center)**: Red tint with red border

### Layout
- Messages use `max-width: 65%` (HOPE/User) or `60%` (3rd party)
- Proper alignment with `align-self` and `margin-left/right: auto`
- Rounded corners differ by position (left-aligned vs right-aligned)

## Testing

To test the new layout:

1. **HOPE Agent Messages**:
   - Select "HOPE v1" in agent selector
   - Send a message
   - Should appear on the left with green styling

2. **3rd Party Model Messages**:
   - Select "Consult: Gemma" or direct model
   - Send a message
   - Should appear centered with blue styling

3. **User Messages**:
   - Type and send any message
   - Should always appear on the right

## Files Modified

1. `continuonbrain/server/static/ui.css` - Added chat message positioning styles
2. `continuonbrain/server/static/chat-overlay.js` - Updated role detection and message rendering

## Future Enhancements

- Add visual indicators (icons) for each message type
- Add animation for message appearance
- Support for multi-turn conversations with proper role tracking
- Better handling of mixed conversations (HOPE + subagent in same turn)
