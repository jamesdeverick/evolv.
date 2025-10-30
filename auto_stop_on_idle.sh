#!/bin/bash
# Auto-stop RunPod after inactivity to save credits
# Usage: Run this in background when starting your pod

IDLE_TIMEOUT=1800  # 30 minutes in seconds (adjust as needed)
CHECK_INTERVAL=60  # Check every 60 seconds

echo "🔍 Auto-stop monitor started"
echo "⏱️  Will stop pod after $((IDLE_TIMEOUT/60)) minutes of no Streamlit connections"

last_activity=$(date +%s)

while true; do
    # Check if Streamlit is running and has active connections
    connections=$(netstat -an | grep ":7860" | grep ESTABLISHED | wc -l)

    current_time=$(date +%s)
    idle_time=$((current_time - last_activity))

    if [ "$connections" -gt 0 ]; then
        # Active connections detected
        last_activity=$current_time
        echo "✅ $(date '+%H:%M:%S') - Active connections: $connections"
    else
        # No connections
        echo "⏳ $(date '+%H:%M:%S') - Idle for $((idle_time/60)) minutes"

        if [ "$idle_time" -gt "$IDLE_TIMEOUT" ]; then
            echo "🛑 STOPPING POD - Exceeded idle timeout of $((IDLE_TIMEOUT/60)) minutes"
            echo "💰 This saved you from burning more credits!"

            # Stop the pod using runpodctl if available
            if command -v runpodctl &> /dev/null; then
                runpodctl stop pod $RUNPOD_POD_ID
            else
                echo "⚠️  runpodctl not found. Please stop pod manually at:"
                echo "   https://www.runpod.io/console/pods"
                # Kill Streamlit and Ollama to at least stop processing
                pkill -f streamlit
                pkill -f ollama
                exit 0
            fi
        fi
    fi

    sleep $CHECK_INTERVAL
done
