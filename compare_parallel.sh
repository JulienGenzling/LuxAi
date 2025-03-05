#!/bin/bash

########################################
# User-defined parameters
########################################
num_runs=100         # How many times to run test.sh
parallel_workers=8    # How many runs to do in parallel at once

########################################
# Prepare temp directory to hold results
########################################
temp_dir=$(mktemp -d)
touch "$temp_dir/done.log"

########################################
# Function to run test.sh once and parse
# its wins_player_0 / wins_player_1
########################################
run_test() {
    local i=$1
    output=$(bash test.sh)

    # Extract wins from the output
    wins_player_0=$(echo "$output" | sed -n "s/.*'player_0': array(\([0-9]*\),.*/\1/p")
    wins_player_1=$(echo "$output" | sed -n "s/.*'player_1': array(\([0-9]*\),.*/\1/p")

    # Store the extracted wins into a temp file
    echo "$wins_player_0 $wins_player_1" > "$temp_dir/result_$i.txt"
}

# Export function and variables so xargs can see them
export -f run_test
export temp_dir

########################################
# 1) Start a background “progress bar” process
########################################
(
  while true; do
    # Count how many lines are in done.log => how many tasks finished
    done_count=$(wc -l < "$temp_dir/done.log")

    # Print a simple progress indicator
    echo -en "\rProgress: $done_count / $num_runs"
    if [[ "$done_count" -eq "$num_runs" ]]; then
      echo
      break
    fi
    sleep 1
  done
) &

progress_pid=$!  # store PID of background progress script

########################################
# 2) Run test.sh multiple times in parallel
#    Then mark each run as “done” by appending to done.log
########################################
seq 1 "$num_runs" | xargs -I{} -P "$parallel_workers" bash -c '
  run_test "$@" && echo 1 >> "'"$temp_dir"'/done.log"
' _ {}

########################################
# 3) Wait for the progress bar to complete
########################################
wait $progress_pid

########################################
# 4) Aggregate final results
########################################
total_series_wins_player_0=0
total_series_wins_player_1=0
total_series_played=0

for file in "$temp_dir"/result_*.txt; do
    if [[ -f "$file" ]]; then
        read w0 w1 < "$file"
        if [[ -n "$w0" && -n "$w1" ]]; then
            if (( w1 > w0 )); then
                total_series_wins_player_1=$((total_series_wins_player_1 + 1))
            elif (( w0 > w1 )); then
                total_series_wins_player_0=$((total_series_wins_player_0 + 1))
            fi
            total_series_played=$((total_series_played + 1))
        fi
    fi
done

########################################
# 5) Output final results
########################################
echo "=================================="
echo "Finished $total_series_played runs."
echo "Player 0 Wins: $total_series_wins_player_0"
echo "Player 1 Wins: $total_series_wins_player_1"
echo "=================================="

# Cleanup temp dir
rm -rf "$temp_dir"