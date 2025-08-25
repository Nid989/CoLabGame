#!/bin/bash

# =============================================================================
# Sequential Experiment Manager
# =============================================================================
# This script runs experiments sequentially in the background
# It's called by start.sh and runs independently
# =============================================================================

# Parameters passed from start.sh
MODEL="$1"
MAX_TOKENS="$2"
PROJECT_DIR="$3"
SCRIPT_DIR="$4"
LOGS_DIR="$5"
OUTPUT_LOGS_DIR="$6"
ERROR_LOGS_DIR="$7"
PID_FILE="$8"
CAFFEINATE_PID_FILE="$9"
shift 9
EXPERIMENTS=("$@")

# Log file for this manager
MANAGER_LOG="$LOGS_DIR/experiment_manager.log"

# Redirect all output to log file
exec > "$MANAGER_LOG" 2>&1

echo "[$(date)] Starting sequential experiment manager"
echo "[$(date)] Manager PID: $$"
echo "[$(date)] Model: $MODEL"
echo "[$(date)] Max Tokens: $MAX_TOKENS"
echo "[$(date)] Sequential Experiments: ${#EXPERIMENTS[@]}"
echo

# Clear any previous completed experiments file
> "${PID_FILE}.completed"

run_experiment() {
    local experiment_name="$1"
    local experiment_index="$2"
    local total_experiments="$3"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_log="$OUTPUT_LOGS_DIR/${experiment_name}_${timestamp}.out"
    local error_log="$ERROR_LOGS_DIR/${experiment_name}_${timestamp}.err"
    
    echo
    echo "============================================================================="
    echo "[$(date)] EXPERIMENT $experiment_index of $total_experiments"
    echo "[$(date)] Starting experiment: $experiment_name"
    echo "[$(date)] Output log: $output_log"
    echo "[$(date)] Error log:  $error_log"
    echo "============================================================================="
    
    # Change to project directory
    cd "$PROJECT_DIR" || {
        echo "[$(date)] ERROR: Failed to change to project directory: $PROJECT_DIR"
        return 1
    }
    
    # Build the command
    local cmd="uv run clem run -g computergame -m $MODEL --max_tokens $MAX_TOKENS -e \"$experiment_name\""
    
    echo "[$(date)] Running command: $cmd"
    echo
    
    # Record experiment start
    local start_time=$(date +%s)
    
    # Run the experiment in background but wait for completion
    nohup bash -c "$cmd" > "$output_log" 2> "$error_log" &
    local experiment_pid=$!
    
    # Update PID file with actual process ID
    echo "RUNNING:$experiment_name:$output_log:$error_log:$start_time:$experiment_pid" > "$PID_FILE"
    
    # Wait for the experiment to complete
    wait $experiment_pid
    local exit_code=$?
    
    # Calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    # Update status based on exit code
    if [[ $exit_code -eq 0 ]]; then
        echo "COMPLETED:$experiment_name:$output_log:$error_log:$duration" >> "${PID_FILE}.completed"
        echo "[$(date)] SUCCESS: Experiment completed successfully in $duration_formatted"
    else
        echo "FAILED:$experiment_name:$output_log:$error_log:$duration:$exit_code" >> "${PID_FILE}.completed"
        echo "[$(date)] ERROR: Experiment failed with exit code $exit_code after $duration_formatted"
    fi
    
    echo "[$(date)] Output logs: $output_log"
    echo "[$(date)] Error logs:  $error_log"
    echo
}

# Run each experiment sequentially
experiment_index=1
total_experiments=${#EXPERIMENTS[@]}
batch_start_time=$(date +%s)

for experiment in "${EXPERIMENTS[@]}"; do
    if [[ -n "$experiment" && ! "$experiment" =~ ^[[:space:]]*# ]]; then
        run_experiment "$experiment" "$experiment_index" "$total_experiments"
        ((experiment_index++))
    fi
done

# Calculate total batch time
batch_end_time=$(date +%s)
batch_duration=$((batch_end_time - batch_start_time))
batch_duration_formatted=$(printf "%02d:%02d:%02d" $((batch_duration/3600)) $((batch_duration%3600/60)) $((batch_duration%60)))

# Stop caffeinate
echo "[$(date)] All experiments completed. Stopping caffeinate..."
if [[ -f "$CAFFEINATE_PID_FILE" ]]; then
    caffeinate_pid=$(cat "$CAFFEINATE_PID_FILE")
    if [[ -n "$caffeinate_pid" ]] && kill -0 "$caffeinate_pid" 2>/dev/null; then
        kill -TERM "$caffeinate_pid" 2>/dev/null || true
        rm -f "$CAFFEINATE_PID_FILE"
        echo "[$(date)] System sleep behavior restored"
    fi
fi

# Clean up current experiment tracking file
rm -f "$PID_FILE"

# Show final summary
echo
echo "============================================================================="
echo "[$(date)] EXPERIMENT BATCH COMPLETED"
echo "============================================================================="
echo "Model: $MODEL"
echo "Max Tokens: $MAX_TOKENS"
echo "Total Experiments: ${#EXPERIMENTS[@]}"
echo

# Show completed experiments summary
completed_count=0
failed_count=0

if [[ -f "${PID_FILE}.completed" ]]; then
    echo "Experiment Results:"
    echo "----------------------------------------------------------------------------"
    
    while IFS=':' read -r status exp_name output_log error_log duration exit_code; do
        if [[ "$status" == "COMPLETED" ]]; then
            duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
            echo "  ✓ $exp_name - Duration: $duration_formatted"
            ((completed_count++))
        elif [[ "$status" == "FAILED" ]]; then
            duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
            echo "  ✗ $exp_name - Duration: $duration_formatted (Exit Code: ${exit_code:-unknown})"
            ((failed_count++))
        fi
    done < "${PID_FILE}.completed"
    
    echo "----------------------------------------------------------------------------"
    echo "Completed: $completed_count | Failed: $failed_count"
else
    echo "No experiments completed (batch was interrupted)"
fi

echo
echo "Log Files:"
echo "  Output Logs: $OUTPUT_LOGS_DIR/"
echo "  Error Logs:  $ERROR_LOGS_DIR/"
echo "  Manager Log: $MANAGER_LOG"
echo "============================================================================="

echo "[$(date)] All experiments completed!"
echo "[$(date)] Total batch time: $batch_duration_formatted"
echo "[$(date)] Sequential experiment manager finished"
