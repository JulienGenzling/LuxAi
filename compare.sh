#!/bin/bash

total_series_wins_player_0=0
total_series_wins_player_1=0
total_series_played=0
bar_width=50  # Width of the bar in characters

# Function to draw the colored bar using tput
draw_bar() {
    local wins_player_0=$1
    local wins_player_1=$2
    local total=$3
    
    # Calculate the proportion of wins
    local prop_player_0=$((wins_player_0 * bar_width / total))
    local prop_player_1=$((wins_player_1 * bar_width / total))
    
    # Draw the bar
    echo -ne "\r"
    tput setab 4  # Blue background for player_0
    for ((i=0; i<prop_player_0; i++)); do echo -n " "; done
    tput setab 1  # Red background for player_1
    for ((i=0; i<prop_player_1; i++)); do echo -n " "; done
    tput sgr0  # Reset colors
    echo -n " $total tests"
}

# Run test.sh 20 times
for i in {1..100}; do
    output=$(bash test.sh)
    
    wins_player_0=$(echo "$output" | sed -n "s/.*'player_0': array(\([0-9]*\),.*/\1/p")
    wins_player_1=$(echo "$output" | sed -n "s/.*'player_1': array(\([0-9]*\),.*/\1/p")
    
    if [[ -n "$wins_player_0" && -n "$wins_player_1" ]]; then
        if (( wins_player_1 > wins_player_0 )); then
            total_series_wins_player_1=$((total_series_wins_player_1 + 1))
        elif (( wins_player_0 > wins_player_1 )); then
            total_series_wins_player_0=$((total_series_wins_player_0 + 1))
        fi
        total_series_played=$((total_series_played + 1))
        
        draw_bar "$total_series_wins_player_0" "$total_series_wins_player_1" "$total_series_played"
    fi
done

echo ""
