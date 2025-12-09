#!/bin/bash
# Test network access to ContinuonBrain API

echo "🧪 Testing ContinuonBrain Network Access"
echo "========================================"
echo ""

# Get robot IP
ROBOT_IP="192.168.68.90"
PORT="8080"
BASE_URL="http://${ROBOT_IP}:${PORT}"

echo "📍 Robot Address: ${BASE_URL}"
echo ""

# Test 1: Ping
echo "1️⃣  Testing network connectivity..."
if ping -c 1 -W 2 ${ROBOT_IP} > /dev/null 2>&1; then
    echo "   ✅ Robot is reachable on network"
else
    echo "   ❌ Cannot reach robot (check network connection)"
    exit 1
fi
echo ""

# Test 2: Port check
echo "2️⃣  Testing port ${PORT}..."
if timeout 3 bash -c "echo > /dev/tcp/${ROBOT_IP}/${PORT}" 2>/dev/null; then
    echo "   ✅ Port ${PORT} is open"
else
    echo "   ❌ Port ${PORT} is not accessible (server may not be running)"
    exit 1
fi
echo ""

# Test 3: HTTP Status
echo "3️⃣  Testing API status endpoint..."
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${BASE_URL}/api/status 2>/dev/null)
if [ "$STATUS_CODE" = "200" ]; then
    echo "   ✅ API is responding (HTTP ${STATUS_CODE})"
else
    echo "   ⚠️  API returned HTTP ${STATUS_CODE} (may still be initializing)"
fi
echo ""

# Test 4: Web UI
echo "4️⃣  Testing Web UI..."
UI_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${BASE_URL}/ui 2>/dev/null)
if [ "$UI_CODE" = "200" ]; then
    echo "   ✅ Web UI is accessible"
    echo "   🌐 Open: ${BASE_URL}/ui"
else
    echo "   ⚠️  Web UI returned HTTP ${UI_CODE}"
fi
echo ""

# Test 5: Chat endpoint
echo "5️⃣  Testing chat endpoint..."
CHAT_RESPONSE=$(curl -s -X POST ${BASE_URL}/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"test"}' 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$CHAT_RESPONSE" ]; then
    echo "   ✅ Chat endpoint is working"
    echo "   Response preview: $(echo $CHAT_RESPONSE | cut -c1-80)..."
else
    echo "   ⚠️  Chat endpoint may not be ready yet"
fi
echo ""

# Test 6: Settings endpoint
echo "6️⃣  Testing settings endpoint..."
SETTINGS_CODE=$(curl -s -o /dev/null -w "%{http_code}" ${BASE_URL}/api/settings 2>/dev/null)
if [ "$SETTINGS_CODE" = "200" ]; then
    echo "   ✅ Settings endpoint is accessible"
else
    echo "   ⚠️  Settings returned HTTP ${SETTINGS_CODE}"
fi
echo ""

# Summary
echo "========================================"
echo "✅ Network Access Test Complete"
echo ""
echo "📱 Flutter App Configuration:"
echo "   Base URL: ${BASE_URL}"
echo "   Status: ${BASE_URL}/api/status"
echo "   Chat: ${BASE_URL}/api/chat"
echo "   Camera: ${BASE_URL}/api/camera/stream"
echo "   Web UI: ${BASE_URL}/ui"
echo ""
echo "🔗 Use these URLs in your Flutter app"
echo "========================================"
