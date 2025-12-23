#!/bin/bash

# =============================================================================
# Experiment Runner Script
# =============================================================================
# This script runs multiple experiments in the background with system awake
# Customize the variables below for your specific experiments
# =============================================================================

# =============================================================================
# CONFIGURATION SECTION - CUSTOMIZE THESE VALUES
# =============================================================================

# Model configuration
MODEL="gpt-5-2025-08-07"
MAX_TOKENS="400"

# Experiment configurations - Add/modify experiments here
# Format: "experiment_name"
EXPERIMENTS=(
    "multi_agent-blackboard-graphic_document-screenshot_a11y_tree-l3"
    "multi_agent-mesh-data_processing-screenshot_a11y_tree-l1"
    "multi_agent-mesh-data_processing-screenshot_a11y_tree-l2"
    "multi_agent-mesh-data_processing-screenshot_a11y_tree-l3"
    "multi_agent-mesh-graphic_document-screenshot_a11y_tree-l1"
    "multi_agent-mesh-graphic_document-screenshot_a11y_tree-l2"
    "multi_agent-mesh-graphic_document-screenshot_a11y_tree-l3"
    "multi_agent-mesh-research_brief-screenshot_a11y_tree-l1"
    "multi_agent-mesh-research_brief-screenshot_a11y_tree-l2"
    "multi_agent-mesh-research_brief-screenshot_a11y_tree-l3"
)

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

# Directories and files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$SCRIPT_DIR/logs"
OUTPUT_LOGS_DIR="$LOGS_DIR/output"
ERROR_LOGS_DIR="$LOGS_DIR/error"
PID_FILE="$SCRIPT_DIR/experiment_pids.txt"
CAFFEINATE_PID_FILE="$SCRIPT_DIR/caffeinate_pid.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# FUNCTIONS
# =============================================================================

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

setup_directories() {
    print_status "Setting up directories..."
    mkdir -p "$LOGS_DIR"
    mkdir -p "$OUTPUT_LOGS_DIR"
    mkdir -p "$ERROR_LOGS_DIR"
    
    # Clear previous PID files
    > "$PID_FILE"
    > "$CAFFEINATE_PID_FILE"
    
    print_success "Directories setup complete"
}

start_caffeinate() {
    print_status "Starting caffeinate to keep system awake..."
    
    # Start caffeinate in background and capture its PID
    caffeinate -d -i &
    local caffeinate_pid=$!
    echo "$caffeinate_pid" > "$CAFFEINATE_PID_FILE"
    
    print_success "System will stay awake (caffeinate PID: $caffeinate_pid)"
}

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
    
    # Change to project directory and run the command
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
    echo "RUNNING:$experiment_name:$output_log:$error_log:$start_time" > "$PID_FILE"
    
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

show_final_summary() {
    echo
    echo "============================================================================="
    echo "[$(date)] EXPERIMENT BATCH COMPLETED"
    echo "============================================================================="
    echo "Model: $MODEL"
    echo "Max Tokens: $MAX_TOKENS"
    echo "Total Experiments: ${#EXPERIMENTS[@]}"
    echo
    
    # Show completed experiments summary
    local completed_count=0
    local failed_count=0
    
    if [[ -f "${PID_FILE}.completed" ]]; then
        echo "Experiment Results:"
        echo "----------------------------------------------------------------------------"
        
        while IFS=':' read -r status exp_name output_log error_log duration exit_code; do
            if [[ "$status" == "COMPLETED" ]]; then
                local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
                echo "  ✓ $exp_name - Duration: $duration_formatted"
                ((completed_count++))
            elif [[ "$status" == "FAILED" ]]; then
                local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
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
    echo "  Monitor:     $SCRIPT_DIR/monitor.sh"
    echo "============================================================================="
}

sequential_experiment_manager() {
    local manager_log="$LOGS_DIR/experiment_manager.log"
    
    # Redirect all manager output to log file
    exec > "$manager_log" 2>&1
    
    echo "[$(date)] Starting sequential experiment manager"
    echo "[$(date)] Manager PID: $$"
    echo "[$(date)] Model: $MODEL"
    echo "[$(date)] Max Tokens: $MAX_TOKENS"
    echo "[$(date)] Sequential Experiments: ${#EXPERIMENTS[@]}"
    echo
    
    # Clear any previous completed experiments file
    > "${PID_FILE}.completed"
    
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
    show_final_summary
    
    echo "[$(date)] All experiments completed successfully!"
    echo "[$(date)] Total batch time: $batch_duration_formatted"
    echo "[$(date)] Sequential experiment manager finished"
}

cleanup_on_exit() {
    echo
    print_warning "Experiment batch interrupted!"
    
    # Stop caffeinate process
    if [[ -f "$CAFFEINATE_PID_FILE" ]]; then
        local caffeinate_pid
        caffeinate_pid=$(cat "$CAFFEINATE_PID_FILE")
        if [[ -n "$caffeinate_pid" ]] && kill -0 "$caffeinate_pid" 2>/dev/null; then
            print_status "Stopping caffeinate process..."
            kill -TERM "$caffeinate_pid" 2>/dev/null || true
            rm -f "$CAFFEINATE_PID_FILE"
        fi
    fi
    
    print_status "System sleep behavior restored"
    
    # Show what was completed before interruption
    show_final_summary
    
    print_warning "Experiment batch was interrupted. Use './scripts/monitor.sh' to check logs."
    
    exit 130
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Set up signal handlers
trap cleanup_on_exit INT TERM

echo "============================================================================="
echo -e "${BLUE}STARTING EXPERIMENT BATCH${NC}"
echo "============================================================================="

# Validate we're in the right directory
if [[ ! -f "$PROJECT_DIR/master.py" ]]; then
    print_error "Cannot find master.py in project directory: $PROJECT_DIR"
    print_error "Please ensure you're running this script from the correct location"
    exit 1
fi

# Setup
setup_directories

# Start keeping system awake
start_caffeinate

print_status "Configuration:"
echo "  Model: $MODEL"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Sequential Experiments: ${#EXPERIMENTS[@]}"
echo

# Start the sequential experiment manager in the background
print_status "Starting sequential experiment manager in background..."

# Use separate script for the manager to avoid complex function passing
nohup "$SCRIPT_DIR/experiment_manager.sh" \
    "$MODEL" \
    "$MAX_TOKENS" \
    "$PROJECT_DIR" \
    "$SCRIPT_DIR" \
    "$LOGS_DIR" \
    "$OUTPUT_LOGS_DIR" \
    "$ERROR_LOGS_DIR" \
    "$PID_FILE" \
    "$CAFFEINATE_PID_FILE" \
    "${EXPERIMENTS[@]}" &

manager_pid=$!

# Store the manager PID for monitoring/stopping
echo "MANAGER:$manager_pid:$(date +%s)" > "${PID_FILE}.manager"

print_success "Sequential experiment manager started!"
print_status "Manager PID: $manager_pid"
print_status "Manager Log: $LOGS_DIR/experiment_manager.log"

echo
echo "============================================================================="
print_status "EXPERIMENT BATCH STARTED IN BACKGROUND"
echo "============================================================================="
echo "The experiments will now run sequentially in the background."
echo "You can safely close this terminal - experiments will continue running."
echo
echo "Monitor Progress:"
echo "  Real-time:     ./scripts/monitor.sh -w"
echo "  Status check:  ./scripts/monitor.sh"
echo "  Follow logs:   ./scripts/monitor.sh -f experiment_name"
echo "  Manager log:   tail -f $LOGS_DIR/experiment_manager.log"
echo
echo "Stop Experiments:"
echo "  Stop all:      ./scripts/stop.sh"
echo
echo "Log Directories:"
echo "  Output logs:   $OUTPUT_LOGS_DIR/"
echo "  Error logs:    $ERROR_LOGS_DIR/"
echo "  Manager log:   $LOGS_DIR/experiment_manager.log"
echo "============================================================================="

print_success "Experiments are now running in background!"
print_status "Use './scripts/monitor.sh' to check progress"

echo
