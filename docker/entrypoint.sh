#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"true"}  # Default if not set
REPO_DIR=${REPO_DIR:-"/workspace/musubi-tuner"}

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS"

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv



if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    echo "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r $REPO_DIR/requirements.txt

    if [ "$DOWNLOAD_MODELS" = "true" ]; then
        echo "DOWNLOAD_MODELS is true, downloading models..."
        MODEL_DIR="/workspace/models"
        mkdir -p "$MODEL_DIR"

        # Download hunyuan-video-t2v-720p
        if [ ! -d "${MODEL_DIR}/hunyuan-video-t2v-720p" ]; then
            huggingface-cli download tencent/HunyuanVideo --local-dir "${MODEL_DIR}"
            # git clone https://huggingface.co/tencent/HunyuanVideo "${MODEL_DIR}" 
            # cd "${MODEL_DIR}"
            # gif lfs pull
            # cd -
        else
            echo "Skipping the model hunyuan-video-t2v-720p download because it already exists."
        fi

        # Download llava-llama-3-8b-text-encoder-tokenizer
        if [ ! -d "${MODEL_DIR}/llava-llama-3-8b-v1_1-transformers" ]; then
            huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir "${MODEL_DIR}/llava-llama-3-8b-v1_1-transformers"
            # git clone https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers "${MODEL_DIR}/llava-llama-3-8b-v1_1-transformers"
            # cd "${MODEL_DIR}/llava-llama-3-8b-v1_1-transformers"
            # git lfs pull
            # cd -
        else
            echo "Skipping the model llava-llama-3-8b-v1_1-transformers download because it already exists."
        fi

        # Preprocess text encoder and tokenizer
        if [ ! -d "${MODEL_DIR}/text_encoder" ]; then
            python $REPO_DIR/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir $MODEL_DIR/llava-llama-3-8b-v1_1-transformers --output_dir "${MODEL_DIR}/text_encoder" 
        else
            echo "Skipping the model text_encoder preprocess because it already exists."
        fi

        # Download clip-vit-large-patch14
        if [ ! -d "${MODEL_DIR}/text_encoder_2" ]; then
            huggingface-cli download openai/clip-vit-large-patch14 --local-dir "${MODEL_DIR}/text_encoder_2"
            # git clone https://huggingface.co/openai/clip-vit-large-patch14 "${MODEL_DIR}/text_encoder_2"
            # cd "${MODEL_DIR}/text_encoder_2"
            # git lfs pull
            # cd -
        else
            echo "Skipping the model clip-vit-large-patch14 download because it already exists."
        fi
        
    else
        echo "DOWNLOAD_MODELS is false, skipping model downloads."
    fi

    # Create marker file
    touch "$INIT_MARKER"
    echo "Initialization complete."
else
    echo "Container already initialized. Skipping first-time setup."
fi

echo "Adding environmnent variables"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export PATH="$REPO_DIR/configs:$PATH"
export PATH="$REPO_DIR:$PATH"

echo $PATH
echo $PYTHONPATH

cd /workspace/musubi-tuner

# Use conda python instead of system python
echo "Starting Gradio interface..."
python gradio_interface.py &

# Use debugpy for debugging
# exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 gradio_interface.py

echo "Starting Tensorboard interface..."
$CONDA_DIR/bin/conda run -n pyenv tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &

wait