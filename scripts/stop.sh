#!/bin/bash

# =============================================================================
# Experiment Stopper Script
# =============================================================================
# This script stops all running experiments and caffeinate processes
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

stop_experiments() {
    local stopped_count=0
    local failed_count=0
    
    print_status "Checking for running experiment manager..."
    
    # First check if manager is running and stop it
    local manager_pid=""
    if [[ -f "${PID_FILE}.manager" ]]; then
        while IFS=':' read -r status pid start_time; do
            if [[ "$status" == "MANAGER" && -n "$pid" ]]; then
                manager_pid="$pid"
                break
            fi
        done < "${PID_FILE}.manager"
        
        if [[ -n "$manager_pid" ]] && kill -0 "$manager_pid" 2>/dev/null; then
            print_status "Found running experiment manager (PID: $manager_pid)"
            print_status "Stopping experiment manager..."
            
            if kill -TERM "$manager_pid" 2>/dev/null; then
                sleep 3
                if kill -0 "$manager_pid" 2>/dev/null; then
                    print_warning "Manager still running, forcing termination..."
                    kill -KILL "$manager_pid" 2>/dev/null
                    sleep 1
                fi
                
                if ! kill -0 "$manager_pid" 2>/dev/null; then
                    print_success "Stopped experiment manager"
                    ((stopped_count++))
                else
                    print_error "Failed to stop experiment manager"
                    ((failed_count++))
                fi
            else
                print_error "Failed to send signal to experiment manager"
                ((failed_count++))
            fi
        else
            print_status "No running experiment manager found"
        fi
    fi
    
    # Check if there's a currently running experiment
    local current_experiment=""
    local current_exp_pid=""
    if [[ -f "$PID_FILE" ]]; then
        while IFS=':' read -r status exp_name output_log error_log start_time exp_pid; do
            if [[ "$status" == "RUNNING" ]]; then
                current_experiment="$exp_name"
                current_exp_pid="$exp_pid"
                print_status "Found running experiment: $exp_name (PID: $exp_pid)"
                break
            fi
        done < "$PID_FILE"
    fi
    
    # Stop current experiment if found
    if [[ -n "$current_exp_pid" ]] && kill -0 "$current_exp_pid" 2>/dev/null; then
        print_status "Stopping current experiment process..."
        
        if kill -TERM "$current_exp_pid" 2>/dev/null; then
            sleep 3
            if kill -0 "$current_exp_pid" 2>/dev/null; then
                print_warning "Experiment process still running, forcing termination..."
                kill -KILL "$current_exp_pid" 2>/dev/null
                sleep 1
            fi
            
            if ! kill -0 "$current_exp_pid" 2>/dev/null; then
                print_success "Stopped experiment: $current_experiment"
                ((stopped_count++))
            else
                print_error "Failed to stop experiment: $current_experiment"
                ((failed_count++))
            fi
        fi
    fi
    
    # Stop any remaining uv/clem processes as backup
    print_status "Checking for any remaining experiment processes..."
    local remaining_pids=$(pgrep -f "uv run clem run" 2>/dev/null || true)
    
    if [[ -n "$remaining_pids" ]]; then
        print_status "Found additional experiment processes, stopping them..."
        echo "$remaining_pids" | while read -r pid; do
            if [[ -n "$pid" ]] && [[ "$pid" != "$current_exp_pid" ]]; then
                print_status "Stopping additional process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 2
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    if [[ -n "$current_experiment" ]]; then
        print_warning "Experiment '$current_experiment' was interrupted and may be incomplete"
    fi
    
    echo
    echo "Experiment Stop Summary:"
    if [[ $stopped_count -gt 0 ]]; then
        echo "  Processes stopped: $stopped_count"
    fi
    if [[ $failed_count -gt 0 ]]; then
        echo "  Failed to stop: $failed_count"
    fi
    if [[ $stopped_count -eq 0 && $failed_count -eq 0 ]]; then
        echo "  No running processes found to stop"
    fi
}

stop_caffeinate() {
    print_status "Stopping caffeinate process..."
    
    if [[ -f "$CAFFEINATE_PID_FILE" ]]; then
        local caffeinate_pid
        caffeinate_pid=$(cat "$CAFFEINATE_PID_FILE")
        
        if [[ -n "$caffeinate_pid" ]] && kill -0 "$caffeinate_pid" 2>/dev/null; then
            if kill -TERM "$caffeinate_pid" 2>/dev/null; then
                print_success "Stopped caffeinate process (PID: $caffeinate_pid)"
            else
                print_error "Failed to stop caffeinate process (PID: $caffeinate_pid)"
            fi
        else
            print_status "Caffeinate process already stopped or not found"
        fi
    else
        print_status "No caffeinate PID file found"
    fi
    
    # Also check for any remaining caffeinate processes we might have started
    local remaining_caffeinate=$(pgrep -f "caffeinate -d -i" 2>/dev/null || true)
    if [[ -n "$remaining_caffeinate" ]]; then
        print_status "Found additional caffeinate processes, stopping them..."
        echo "$remaining_caffeinate" | while read -r pid; do
            if [[ -n "$pid" ]]; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
}

cleanup_files() {
    print_status "Cleaning up tracking files..."
    
    if [[ -f "$PID_FILE" ]]; then
        rm "$PID_FILE"
        print_success "Removed experiment tracking file"
    fi
    
    if [[ -f "${PID_FILE}.manager" ]]; then
        rm "${PID_FILE}.manager"
        print_success "Removed manager tracking file"
    fi
    
    if [[ -f "${PID_FILE}.completed" ]]; then
        print_status "Preserving completed experiments log: ${PID_FILE}.completed"
    fi
    
    if [[ -f "$CAFFEINATE_PID_FILE" ]]; then
        rm "$CAFFEINATE_PID_FILE"
        print_success "Removed caffeinate PID file"
    fi
}

show_final_status() {
    echo
    echo "============================================================================="
    echo -e "${GREEN}STOP OPERATION COMPLETE${NC}"
    echo "============================================================================="
    
    # Check if any related processes are still running
    local remaining_uv=$(pgrep -f "uv run clem" 2>/dev/null | wc -l)
    local remaining_caffeinate=$(pgrep -f "caffeinate -d -i" 2>/dev/null | wc -l)
    
    if [[ $remaining_uv -eq 0 && $remaining_caffeinate -eq 0 ]]; then
        print_success "All processes have been stopped successfully"
    else
        if [[ $remaining_uv -gt 0 ]]; then
            print_warning "$remaining_uv experiment processes may still be running"
        fi
        if [[ $remaining_caffeinate -gt 0 ]]; then
            print_warning "$remaining_caffeinate caffeinate processes may still be running"
        fi
        echo
        print_status "You can check remaining processes with:"
        echo "  ps aux | grep -E '(uv run clem|caffeinate -d -i)'"
    fi
    
    echo
    print_status "Your system will now be able to sleep normally"
    echo "============================================================================="
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "============================================================================="
echo -e "${RED}STOPPING ALL EXPERIMENTS${NC}"
echo "============================================================================="

# Parse command line arguments
FORCE_MODE=false
if [[ "$1" == "--force" || "$1" == "-f" ]]; then
    FORCE_MODE=true
    print_warning "Force mode enabled - will use SIGKILL immediately"
fi

# Stop experiments
stop_experiments

# Stop caffeinate
stop_caffeinate

# Clean up files
cleanup_files

# Show final status
show_final_status

print_success "Stop operation completed!"

# If there were any issues, suggest manual cleanup
echo
print_status "If any processes are still running, you can manually check with:"
echo "  ps aux | grep -E '(uv run clem|caffeinate)'"
echo "  pgrep -f 'uv run clem'"
