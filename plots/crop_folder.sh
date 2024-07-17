#!/bin/bash

# Loop through all PDF files in fig2 folder
for file in *.pdf; do
    # Get filename without extension
    filename=$(basename "$file" .pdf)
    
    # Perform cropping using pdfcrop
    pdfcrop "$file" "$filename-cropped.pdf"
    
    # Print status
    echo "Cropped $file"
done

echo "All PDFs cropped."

