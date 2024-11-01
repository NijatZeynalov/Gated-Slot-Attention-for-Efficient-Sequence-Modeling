#!/bin/bash

# Default values for training parameters
MODEL_SIZE="7b"
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=4
WARMUP_STEPS=100
OUTPUT_DIR="outputs"
DATA_DIR="data"
MODEL_TYPE="llama2"
USE_FP16=true
MAX_SEQ_LENGTH=2048
NUM_SLOTS=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --gradient_accumulation)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
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
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --no_fp16)
            USE_FP16=false
            shift
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --num_slots)
            NUM_SLOTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model_size             Model size (default: 7b)"
            echo "  --batch_size             Batch size (default: 32)"
            echo "  --learning_rate          Learning rate (default: 2e-5)"
            echo "  --num_epochs             Number of epochs (default: 3)"
            echo "  --gradient_accumulation  Gradient accumulation steps (default: 4)"
            echo "  --warmup_steps           Warmup steps (default: 100)"
            echo "  --output_dir            Output directory (default: outputs)"
            echo "  --data_dir              Data directory (default: data)"
            echo "  --model_type            Model type (default: llama2)"
            echo "  --no_fp16               Disable FP16 training"
            echo "  --max_seq_length        Maximum sequence length (default: 2048)"
            echo "  --num_slots             Number of GSA slots (default: 64)"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Log training parameters
echo "Starting training with following parameters:"
echo "Model Size: $MODEL_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Model Type: $MODEL_TYPE"
echo "FP16 Training: $USE_FP16"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "Number of GSA Slots: $NUM_SLOTS"

# Set up GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Modify according to available GPUs

# Run the training script
python -m src.main \
    --mode train \
    --model_size "$MODEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL_TYPE" \
    --use_fp16 "$USE_FP16" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --num_slots "$NUM_SLOTS" \
    --log_dir "${OUTPUT_DIR}/logs" \
    --checkpoint_dir "${OUTPUT_DIR}/checkpoints" \
    2>&1 | tee "${OUTPUT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).log"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with error code $?"
    exit 1
fi