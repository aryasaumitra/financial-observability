#!/bin/bash

# Check if company name was provided
if [ -z "$1" ]; then
    echo "Usage: ./add-company.sh <company-name>"
    exit 1
fi

COMPANY=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

BASE_DIR="companies/$COMPANY"

# Create directory structure
mkdir -p "$BASE_DIR/annual-reports"
mkdir -p "$BASE_DIR/earning-calls"
mkdir -p "$BASE_DIR/others"

echo "Created company: $COMPANY"
echo
echo "$BASE_DIR/"
echo "├── annual-reports/"
echo "├── earning-calls/"
echo "└── others/"