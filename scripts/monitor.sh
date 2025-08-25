#!/bin/bash

# =============================================================================
# Experiment Monitor Script
# =============================================================================
# This script monitors running experiments and shows their status
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/experiment_pids.txt"
CAFFEINATE_PID_FILE="$SCRIPT_DIR/caffeinate_pid.txt"
LOGS_DIR="$SCRIPT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
DEFAULT_LOG_LINES=20
DEFAULT_REFRESH_INTERVAL=5

# =============================================================================
# FUNCTIONS
# =============================================================================

print_header() {
    echo -e "${CYAN}$1${NC}"
}

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

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -l, --logs [NUM]        Show last NUM lines of logs (default: $DEFAULT_LOG_LINES)"
    echo "  -w, --watch [INTERVAL]  Watch mode - refresh every INTERVAL seconds (default: $DEFAULT_REFRESH_INTERVAL)"
    echo "  -s, --summary           Show only summary (no logs)"
    echo "  -f, --follow EXP_NAME   Follow logs for specific experiment"
    echo "  --status                Show process status only"
    echo "  --system                Show system resources"
    echo
    echo "Examples:"
    echo "  $0                      Show status and recent logs"
    echo "  $0 -w                   Watch mode with 5-second refresh"
    echo "  $0 -w 10                Watch mode with 10-second refresh"
    echo "  $0 -l 50                Show last 50 lines of each log"
    echo "  $0 -s                   Show only summary"
    echo "  $0 -f single_agent-chrome  Follow logs for specific experiment"
}

get_system_info() {
    local cpu_usage
    local memory_usage
    local load_avg
    
    # Get CPU and memory info (macOS specific)
    cpu_usage=$(top -l 1 -s 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
    memory_usage=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
    load_avg=$(uptime | awk -F'load averages: ' '{print $2}')
    
    echo "System Resources:"
    echo "  Load Average: $load_avg"
    if [[ -n "$cpu_usage" ]]; then
        echo "  CPU Usage: ${cpu_usage}%"
    fi
    echo "  Uptime: $(uptime | awk -F'up ' '{print $2}' | awk -F', [0-9]* user' '{print $1}')"
}

get_process_status() {
    local running_count=0
    local completed_count=0
    local failed_count=0
    local manager_running=false
    
    print_header "EXPERIMENT STATUS"
    echo "============================================================================="
    
    # Check manager status first
    if [[ -f "${PID_FILE}.manager" ]]; then
        while IFS=':' read -r status manager_pid start_time; do
            if [[ "$status" == "MANAGER" && -n "$manager_pid" ]]; then
                if kill -0 "$manager_pid" 2>/dev/null; then
                    manager_running=true
                    local current_time=$(date +%s)
                    local elapsed=$((current_time - start_time))
                    local elapsed_formatted=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
                    echo -e "${BLUE}Manager Status: ${GREEN}RUNNING${NC} (PID: $manager_pid, Runtime: $elapsed_formatted)${NC}"
                else
                    echo -e "${BLUE}Manager Status: ${RED}STOPPED${NC} (PID: $manager_pid)${NC}"
                fi
            fi
        done < "${PID_FILE}.manager"
    fi
    
    if [[ "$manager_running" == false ]]; then
        echo -e "${BLUE}Manager Status: ${YELLOW}NOT FOUND${NC}${NC}"
    fi
    
    echo
    printf "%-40s %-12s %s\n" "EXPERIMENT" "STATUS" "DURATION/RUNTIME"
    echo "----------------------------------------------------------------------------"
    
    # Check for currently running experiment
    if [[ -f "$PID_FILE" ]]; then
        while IFS=':' read -r status exp_name output_log error_log start_time exp_pid; do
            if [[ "$status" == "RUNNING" && -n "$exp_name" ]]; then
                local current_time=$(date +%s)
                local elapsed=$((current_time - start_time))
                local elapsed_formatted=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
                
                # Check if the experiment process is actually still running
                local process_status="RUNNING"
                if [[ -n "$exp_pid" ]] && ! kill -0 "$exp_pid" 2>/dev/null; then
                    process_status="FINISHING"
                fi
                
                if [[ "$process_status" == "RUNNING" ]]; then
                    printf "%-40s ${YELLOW}%-12s${NC} %s\n" "${exp_name:0:39}" "RUNNING" "$elapsed_formatted"
                else
                    printf "%-40s ${CYAN}%-12s${NC} %s\n" "${exp_name:0:39}" "FINISHING" "$elapsed_formatted"
                fi
                ((running_count++))
            fi
        done < "$PID_FILE"
    fi
    
    # Check for completed experiments
    if [[ -f "${PID_FILE}.completed" ]]; then
        while IFS=':' read -r status exp_name output_log error_log duration exit_code; do
            if [[ "$status" == "COMPLETED" ]]; then
                local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
                printf "%-40s ${GREEN}%-12s${NC} %s\n" "${exp_name:0:39}" "COMPLETED" "$duration_formatted"
                ((completed_count++))
            elif [[ "$status" == "FAILED" ]]; then
                local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $((duration%3600/60)) $((duration%60)))
                printf "%-40s ${RED}%-12s${NC} %s (Exit: ${exit_code:-?})\n" "${exp_name:0:39}" "FAILED" "$duration_formatted"
                ((failed_count++))
            fi
        done < "${PID_FILE}.completed"
    fi
    
    local total_count=$((running_count + completed_count + failed_count))
    
    if [[ $total_count -eq 0 ]]; then
        if [[ "$manager_running" == true ]]; then
            print_warning "Manager is running but no experiments started yet."
        else
            print_warning "No experiments found. Run './scripts/start.sh' to begin experiments."
        fi
        echo
        return 1
    fi
    
    echo "----------------------------------------------------------------------------"
    echo "Total: $total_count | Running: $running_count | Completed: $completed_count | Failed: $failed_count"
    
    if [[ "$manager_running" == true && $running_count -eq 0 && $total_count -gt 0 ]]; then
        echo -e "${YELLOW}Note: Manager is running - more experiments may start soon.${NC}"
    elif [[ "$manager_running" == false && $running_count -gt 0 ]]; then
        echo -e "${YELLOW}Note: Experiments running but manager stopped - may be interrupted.${NC}"
    fi
    
    echo
    
    return 0
}

get_caffeinate_status() {
    local caffeinate_running=false
    
    if [[ -f "$CAFFEINATE_PID_FILE" ]]; then
        local caffeinate_pid
        caffeinate_pid=$(cat "$CAFFEINATE_PID_FILE")
        
        if [[ -n "$caffeinate_pid" ]] && kill -0 "$caffeinate_pid" 2>/dev/null; then
            local runtime=$(ps -o etime= -p "$caffeinate_pid" 2>/dev/null | xargs)
            echo -e "System Sleep: ${GREEN}DISABLED${NC} (caffeinate PID: $caffeinate_pid, Runtime: $runtime)"
            caffeinate_running=true
        fi
    fi
    
    if [[ "$caffeinate_running" == false ]]; then
        echo -e "System Sleep: ${YELLOW}ENABLED${NC} (no caffeinate process found)"
    fi
    echo
}

show_experiment_logs() {
    local num_lines="${1:-$DEFAULT_LOG_LINES}"
    local specific_exp="$2"
    
    local logs_shown=0
    
    # Function to show logs for a specific experiment
    show_logs_for_experiment() {
        local exp_name="$1"
        local output_log="$2"
        local error_log="$3"
        local status="$4"
        
        # If specific experiment requested, skip others
        if [[ -n "$specific_exp" ]] && [[ ! "$exp_name" =~ $specific_exp ]]; then
            return
        fi
        
        # Show output logs
        if [[ -f "$output_log" ]]; then
            print_header "OUTPUT LOGS: $exp_name (last $num_lines lines)"
            echo "Output log: $output_log"
            echo "Status: $status"
            echo "----------------------------------------------------------------------------"
            tail -n "$num_lines" "$output_log" 2>/dev/null || echo "Unable to read output log file"
            echo
            ((logs_shown++))
        fi
        
        # Show error logs
        if [[ -f "$error_log" ]]; then
            print_header "ERROR LOGS: $exp_name (last $num_lines lines)"
            echo "Error log: $error_log"
            echo "Status: $status"
            echo "----------------------------------------------------------------------------"
            local error_content
            error_content=$(tail -n "$num_lines" "$error_log" 2>/dev/null)
            if [[ -n "$error_content" ]]; then
                echo "$error_content"
            else
                echo -e "${GREEN}[No error output]${NC}"
            fi
            echo
            echo
            ((logs_shown++))
        fi
        
        # If neither log file exists for specific experiment
        if [[ -n "$specific_exp" ]] && [[ ! -f "$output_log" ]] && [[ ! -f "$error_log" ]]; then
            print_error "Log files not found: $output_log, $error_log"
        fi
    }
    
    # Check currently running experiment
    if [[ -f "$PID_FILE" ]]; then
        while IFS=':' read -r status exp_name output_log error_log start_time exp_pid; do
            if [[ "$status" == "RUNNING" && -n "$exp_name" ]]; then
                local process_status="RUNNING"
                if [[ -n "$exp_pid" ]] && ! kill -0 "$exp_pid" 2>/dev/null; then
                    process_status="FINISHING"
                fi
                
                if [[ "$process_status" == "RUNNING" ]]; then
                    show_logs_for_experiment "$exp_name" "$output_log" "$error_log" "${YELLOW}[RUNNING]${NC}"
                else
                    show_logs_for_experiment "$exp_name" "$output_log" "$error_log" "${CYAN}[FINISHING]${NC}"
                fi
            fi
        done < "$PID_FILE"
    fi
    
    # Check completed experiments  
    if [[ -f "${PID_FILE}.completed" ]]; then
        while IFS=':' read -r status exp_name output_log error_log duration exit_code; do
            if [[ "$status" == "COMPLETED" ]]; then
                show_logs_for_experiment "$exp_name" "$output_log" "$error_log" "${GREEN}[COMPLETED]${NC}"
            elif [[ "$status" == "FAILED" ]]; then
                show_logs_for_experiment "$exp_name" "$output_log" "$error_log" "${RED}[FAILED - Exit: ${exit_code:-?}]${NC}"
            fi
        done < "${PID_FILE}.completed"
    fi
    
    if [[ $logs_shown -eq 0 ]]; then
        if [[ -n "$specific_exp" ]]; then
            print_warning "No logs found for experiment matching: $specific_exp"
        else
            print_warning "No log files found"
        fi
    fi
}

follow_logs() {
    local exp_pattern="$1"
    local output_log=""
    local error_log=""
    local exp_name=""
    
    # Search in currently running experiment
    if [[ -f "$PID_FILE" ]]; then
        while IFS=':' read -r status experiment_name output_path error_path start_time exp_pid; do
            if [[ "$status" == "RUNNING" && "$experiment_name" =~ $exp_pattern ]]; then
                output_log="$output_path"
                error_log="$error_path"
                exp_name="$experiment_name"
                print_status "Following logs for currently running: $exp_name"
                break
            fi
        done < "$PID_FILE"
    fi
    
    # If not found in running, search in completed experiments
    if [[ -z "$output_log" && -f "${PID_FILE}.completed" ]]; then
        while IFS=':' read -r status experiment_name output_path error_path duration exit_code; do
            if [[ "$experiment_name" =~ $exp_pattern ]]; then
                output_log="$output_path"
                error_log="$error_path"
                exp_name="$experiment_name"
                if [[ "$status" == "COMPLETED" ]]; then
                    print_status "Following logs for completed experiment: $exp_name"
                else
                    print_status "Following logs for failed experiment: $exp_name"
                fi
                break
            fi
        done < "${PID_FILE}.completed"
    fi
    
    if [[ -z "$output_log" ]]; then
        print_error "No experiment found matching: $exp_pattern"
        echo
        echo "Available experiments:"
        
        # List running experiments
        if [[ -f "$PID_FILE" ]]; then
            while IFS=':' read -r status experiment_name output_path error_path start_time exp_pid; do
                if [[ "$status" == "RUNNING" ]]; then
                    echo -e "  - ${YELLOW}$experiment_name${NC} (currently running)"
                fi
            done < "$PID_FILE"
        fi
        
        # List completed experiments
        if [[ -f "${PID_FILE}.completed" ]]; then
            while IFS=':' read -r status experiment_name output_path error_path duration exit_code; do
                if [[ "$status" == "COMPLETED" ]]; then
                    echo -e "  - ${GREEN}$experiment_name${NC} (completed)"
                elif [[ "$status" == "FAILED" ]]; then
                    echo -e "  - ${RED}$experiment_name${NC} (failed)"
                fi
            done < "${PID_FILE}.completed"
        fi
        
        return 1
    fi
    
    print_status "Output log: $output_log"
    print_status "Error log:  $error_log"
    
    # Check if at least one log file exists
    if [[ ! -f "$output_log" ]] && [[ ! -f "$error_log" ]]; then
        print_error "No log files found for: $exp_name"
        return 1
    fi
    
    echo "Press Ctrl+C to stop following..."
    echo "============================================================================="
    
    # Follow both logs simultaneously with labels
    if [[ -f "$output_log" ]] && [[ -f "$error_log" ]]; then
        print_status "Following both output and error logs..."
        echo
        tail -f "$output_log" "$error_log" | while read -r line; do
            # Add some visual distinction between output and error logs
            if [[ "$line" =~ ^\=\=\> ]]; then
                # This is a filename header from tail -f
                if [[ "$line" =~ \.out ]]; then
                    echo -e "${GREEN}$line${NC}"
                elif [[ "$line" =~ \.err ]]; then
                    echo -e "${RED}$line${NC}"
                else
                    echo "$line"
                fi
            else
                echo "$line"
            fi
        done
    elif [[ -f "$output_log" ]]; then
        print_status "Following output log only..."
        tail -f "$output_log"
    elif [[ -f "$error_log" ]]; then
        print_status "Following error log only..."
        tail -f "$error_log"
    fi
}

monitor_once() {
    local show_logs="$1"
    local num_lines="$2"
    local show_system="$3"
    
    clear
    echo "============================================================================="
    echo -e "${PURPLE}EXPERIMENT MONITOR${NC} - $(date)"
    echo "============================================================================="
    echo
    
    # Show system info if requested
    if [[ "$show_system" == "true" ]]; then
        get_system_info
        echo
    fi
    
    # Show caffeinate status
    get_caffeinate_status
    
    # Show process status
    if ! get_process_status; then
        return 1
    fi
    
    # Show logs if requested
    if [[ "$show_logs" == "true" ]]; then
        show_experiment_logs "$num_lines"
    fi
    
    return 0
}

watch_mode() {
    local interval="$1"
    local show_logs="$2"
    local num_lines="$3"
    local show_system="$4"
    
    print_status "Starting watch mode (refresh every ${interval}s). Press Ctrl+C to exit."
    
    while true; do
        if ! monitor_once "$show_logs" "$num_lines" "$show_system"; then
            sleep "$interval"
            continue
        fi
        
        echo
        echo -e "${BLUE}[Next refresh in ${interval}s - Press Ctrl+C to exit]${NC}"
        sleep "$interval"
    done
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Parse command line arguments
SHOW_LOGS=true
SHOW_SUMMARY=false
WATCH_MODE=false
FOLLOW_MODE=false
STATUS_ONLY=false
SHOW_SYSTEM=false
NUM_LINES="$DEFAULT_LOG_LINES"
REFRESH_INTERVAL="$DEFAULT_REFRESH_INTERVAL"
FOLLOW_EXPERIMENT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--logs)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                NUM_LINES="$2"
                shift
            fi
            SHOW_LOGS=true
            shift
            ;;
        -w|--watch)
            WATCH_MODE=true
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                REFRESH_INTERVAL="$2"
                shift
            fi
            shift
            ;;
        -s|--summary)
            SHOW_SUMMARY=true
            SHOW_LOGS=false
            shift
            ;;
        -f|--follow)
            FOLLOW_MODE=true
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                FOLLOW_EXPERIMENT="$2"
                shift
            fi
            shift
            ;;
        --status)
            STATUS_ONLY=true
            SHOW_LOGS=false
            shift
            ;;
        --system)
            SHOW_SYSTEM=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Handle different modes
if [[ "$FOLLOW_MODE" == true ]]; then
    if [[ -z "$FOLLOW_EXPERIMENT" ]]; then
        print_error "Please specify an experiment name pattern to follow"
        exit 1
    fi
    follow_logs "$FOLLOW_EXPERIMENT"
elif [[ "$WATCH_MODE" == true ]]; then
    watch_mode "$REFRESH_INTERVAL" "$SHOW_LOGS" "$NUM_LINES" "$SHOW_SYSTEM"
else
    monitor_once "$SHOW_LOGS" "$NUM_LINES" "$SHOW_SYSTEM"
fi
