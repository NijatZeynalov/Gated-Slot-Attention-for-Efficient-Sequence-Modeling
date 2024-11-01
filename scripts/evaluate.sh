#!/bin/bash

# Default values for evaluation parameters
MODEL_PATH=""
BATCH_SIZE=32
OUTPUT_DIR="eval_outputs"
DATA_DIR="data"
EVAL_SPLIT="test"
MAX_SEQ_LENGTH=2048
NUM_SLOTS=64
METRICS="perplexity,accuracy"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --eval_split)
            EVAL_SPLIT="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --num_slots)
            NUM_SLOTS="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model_path      Path to the trained model (required)"
            echo "  --batch_size      Batch size for evaluation (default: 32)"
            echo "  --output_dir      Output directory (default: eval_outputs)"
            echo "  --data_dir        Data directory (default: data)"
            echo "  --eval_split      Evaluation split (default: test)"
            echo "  --max_seq_length  Maximum sequence length (default: 2048)"
            echo "  --num_slots       Number of GSA slots (default: 64)"
            echo "  --metrics         Comma-separated list of metrics to evaluate (default: perplexity,accuracy)"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Log evaluation parameters
echo "Starting evaluation with following parameters:"
echo "Model Path: $MODEL_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Evaluation Split: $EVAL_SPLIT"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "Number of GSA Slots: $NUM_SLOTS"
echo "Metrics: $METRICS"

# Set up GPU configuration
export CUDA_VISIBLE_DEVICES=0  # Modify according to available GPUs

# Run the evaluation script
python -m src.main \
    --mode evaluate \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --eval_split "$EVAL_SPLIT" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --num_slots "$NUM_SLOTS" \
    --metrics "$METRICS" \
    2>&1 | tee "${OUTPUT_DIR}/eval_log_$(date +%Y%m%d_%H%M%S).log"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Evaluation failed with error code $?"
    exit 1
fi

# Generate evaluation summary
echo "Generating evaluation summary..."
python -m src.utils.visualizer \
    --results_dir "$OUTPUT_DIR" \
    --output_file "${OUTPUT_DIR}/evaluation_summary.pdf"

if [ $? -eq 0 ]; then
    echo "Evaluation summary generated: ${OUTPUT_DIR}/evaluation_summary.pdf"
else
    echo "Failed to generate evaluation summary"
fi