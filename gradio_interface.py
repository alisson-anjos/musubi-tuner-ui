import queue
import signal
import subprocess
import threading
import gradio as gr
import os
from datetime import datetime, timedelta
import json
import toml
import shutil
import zipfile
import tempfile
import time
from pathlib import Path
from config_classes import GeneralConfig, ImageDataset, VideoDataset, TrainingConfig, DatasetConfig

DEVELOPMENT = False

# Determine if running on container by checking the environment variable
IS_CONTAINER = os.getenv("IS_CONTAINER", "false").lower() == "true"

# Working directories
MODEL_DIR = "/workspace/models" if IS_CONTAINER else os.path.join(os.getcwd(), "models")
DATASET_DIR = "/workspace/datasets" if IS_CONTAINER else os.path.join(os.getcwd(), "datasets") 
OUTPUT_DIR = "/workspace/outputs" if IS_CONTAINER else os.path.join(os.getcwd(), "outputs")
CONFIG_DIR = "/workspace/configs" if IS_CONTAINER else os.path.join(os.getcwd(), "configs") 

# Maximum number of media to display in the gallery
MAX_MEDIA = 50

CONDA_DIR = os.getenv("CONDA_DIR", "/opt/conda")  # Directory where Conda is installed in the Docker container

# Maximum upload size in MB (Gradio expects max_file_size in MB)
MAX_UPLOAD_SIZE_MB = 500 if IS_CONTAINER else None  # 500MB or no limit

# Create directories if they don't exist
for dir_path in [MODEL_DIR, DATASET_DIR, OUTPUT_DIR, CONFIG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

process_dict = {}
process_lock = threading.Lock()

log_queue = queue.Queue()

def read_subprocess_output(proc, log_queue):
    for line in iter(proc.stdout.readline, b''):
        decoded_line = line.decode('utf-8')
        log_queue.put(decoded_line)
    proc.stdout.close()
    proc.wait()
    with process_lock:
        pid = proc.pid
        if pid in process_dict:
            del process_dict[pid]

def update_logs(log_box, subprocess_proc):
    new_logs = ""
    while not log_queue.empty():
        new_logs += log_queue.get()
    return log_box + new_logs

def clear_logs():
    while not log_queue.empty():
        log_queue.get()
    return ""

def create_dataset_config(dataset_path: str,
                        config_dir: str,
                        num_repeats: int, 
                        resolutions: list, 
                        enable_ar_bucket: bool, 
                        min_ar: float, 
                        max_ar: float, 
                        num_ar_buckets: int, 
                        frame_buckets: list,
                        ar_buckets: list) -> str:
    """Create and save the dataset configuration in TOML format."""
    dataset_config = {
        "resolutions": resolutions,
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": frame_buckets,
        "ar_buckets": ar_buckets,	
        "directory": [
            {
                "path": dataset_path,
                "num_repeats": num_repeats
            }
        ]
    }
    dataset_file = f"dataset_config.toml"
    dataset_path_full = os.path.join(config_dir, dataset_file)
    with open(dataset_path_full, "w") as f:
        toml.dump(dataset_config, f)
    return dataset_path_full

def get_datasets():
    datasets = []
    for dataset in os.listdir(DATASET_DIR):
        datasets.append(dataset)
    return datasets

def load_training_config(dataset_name):
    training_config_path = os.path.join(CONFIG_DIR, dataset_name, "training_config.toml")
    dataset_config_path = os.path.join(CONFIG_DIR, dataset_name, "dataset_config.toml")
    
    config = {}
    
    # Load training configuration
    if not os.path.exists(training_config_path):
        return None, f"Training configuration file not found for the dataset '{dataset_name}'."
    
    try:
        with open(training_config_path, "r") as f:
            training_config = toml.load(f)
        config.update(training_config)
    except Exception as e:
        return None, f"Error loading training configuration: {str(e)}"
    
    # Load dataset configuration
    if not os.path.exists(dataset_config_path):
        return None, f"Dataset configuration file not found for the dataset '{dataset_name}'."
    
    try:
        with open(dataset_config_path, "r") as f:
            dataset_config = toml.load(f)
        config["dataset"] = dataset_config
    except Exception as e:
        return None, f"Error loading dataset configuration: {str(e)}"
    
    return config, None

def extract_config_values(config):
    """
    Extracts training parameters from the configuration dictionary.

    Args:
        config (dict): Dictionary containing training configurations.

    Returns:
        dict: Dictionary with the extracted values.
    """
    training_params = config.get("epochs", 1000)
    batch_size = config.get("micro_batch_size_per_gpu", 1)
    lr = config.get("optimizer", {}).get("lr", 2e-5)
    save_every = config.get("save_every_n_epochs", 2)
    eval_every = config.get("eval_every_n_epochs", 1)
    rank = config.get("adapter", {}).get("rank", 32)
    only_double_blocks = config.get("adapter", {}).get("only_double_blocks", False)
    dtype = config.get("adapter", {}).get("dtype", "bfloat16")
    transformer_path = config.get("model", {}).get("transformer_path", "/workspace/models/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors")
    vae_path = config.get("model", {}).get("vae_path", "/workspace/models/hunyuan_video_vae_fp32.safetensors")
    llm_path = config.get("model", {}).get("llm_path", "/workspace/models/llava-llama-3-8b-text-encoder-tokenizer")
    clip_path = config.get("model", {}).get("clip_path", "/workspace/models/clip-vit-large-patch14")
    optimizer_type = config.get("optimizer", {}).get("type", "adamw_optimi")
    betas = config.get("optimizer", {}).get("betas", [0.9, 0.99])
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.01)
    eps = config.get("optimizer", {}).get("eps", 1e-8)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    num_repeats = config.get('dataset', {}).get('directory', [{}])[:1][0].get('num_repeats', 10)
    resolutions = config.get("dataset", {}).get("resolutions", [512])
    enable_ar_bucket = config.get("dataset", {}).get("enable_ar_bucket", True)
    min_ar = config.get("dataset", {}).get("min_ar", 0.5)
    max_ar = config.get("dataset", {}).get("max_ar", 2.0)
    num_ar_buckets = config.get("dataset", {}).get("num_ar_buckets", 7)
    ar_buckets = config.get("dataset", {}).get("ar_buckets", None)
    frame_buckets = config.get("dataset", {}).get("frame_buckets", [1, 33, 65])
    gradient_clipping = config.get("gradient_clipping", 1.0)
    warmup_steps = config.get("warmup_steps", 100)
    eval_before_first_step = config.get("eval_before_first_step", True)
    eval_micro_batch_size_per_gpu = config.get("eval_micro_batch_size_per_gpu", 1)
    eval_gradient_accumulation_steps = config.get("eval_gradient_accumulation_steps", 1)
    checkpoint_every_n_minutes = config.get("checkpoint_every_n_minutes", 120)
    activation_checkpointing = config.get("activation_checkpointing", True)
    partition_method = config.get("partition_method", "parameters")
    save_dtype = config.get("save_dtype", "bfloat16")
    caching_batch_size = config.get("caching_batch_size", 1)
    steps_per_print = config.get("steps_per_print", 1)
    video_clip_mode = config.get("video_clip_mode", "single_middle")
    
    # Convert lists to JSON strings to fill text fields
    betas_str = json.dumps(betas)
    resolutions_str = json.dumps(resolutions)
    frame_buckets_str = json.dumps(frame_buckets)
    ar_buckets_str = json.dumps(ar_buckets) if ar_buckets else ""
    
    return {
        "epochs": training_params,
        "batch_size": batch_size,
        "lr": lr,
        "save_every": save_every,
        "eval_every": eval_every,
        "rank": rank,
        "only_double_blocks": only_double_blocks,
        "dtype": dtype,
        "transformer_path": transformer_path,
        "vae_path": vae_path,
        "llm_path": llm_path,
        "clip_path": clip_path,
        "optimizer_type": optimizer_type,
        "betas": betas_str,
        "weight_decay": weight_decay,
        "eps": eps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_repeats": num_repeats,
        "resolutions_input": resolutions_str,
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "ar_buckets": ar_buckets_str,
        "frame_buckets": frame_buckets_str,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "eval_before_first_step": eval_before_first_step,
        "eval_micro_batch_size_per_gpu": eval_micro_batch_size_per_gpu,
        "eval_gradient_accumulation_steps": eval_gradient_accumulation_steps,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": activation_checkpointing,
        "partition_method": partition_method,
        "save_dtype": save_dtype,
        "caching_batch_size": caching_batch_size,
        "steps_per_print": steps_per_print,
        "video_clip_mode": video_clip_mode
    }

def validate_resolutions(resolutions):
    try:
        if resolutions == "":
            return None, None
        
        # Attempt to parse the input as JSON
        resolutions_list = json.loads(resolutions)
        
        # Check if the parsed object is a list
        if not isinstance(resolutions_list, list):
            return "Error: resolutions must be a list.", None
        
        # Case 1: List of numbers (int or float)
        if all(isinstance(b, (int, float)) for b in resolutions_list):
            return None, resolutions_list
        
        # Case 2: List of lists of numbers
        elif all(
            isinstance(sublist, list) and all(isinstance(item, (int, float)) for item in sublist)
            for sublist in resolutions_list
        ):
            return None, resolutions_list
        
        else:
            return (
                "Error: resolutions must be a list of numbers or a list of lists of numbers. "
                "Valid examples: [512] or [512, 768, 1024] or [[512, 512], [1280, 720]]"
            ), None
    
    except json.JSONDecodeError as e:
        return f"Error parsing resolutions: {str(e)}", None
    except Exception as e:
        return f"Unexpected error while validating resolutions: {str(e)}", None
    
def validate_target_frames(target_frames):
    try:
        # Attempt to parse the input as JSON
        target_frames_list = json.loads(target_frames)
        
        # Check if the parsed object is a list
        if not isinstance(target_frames_list, list):
            return "Error: Target Frames must be a list.", None
        
        # Case 1: List of numbers (int or float)
        if all(isinstance(b, int) for b in target_frames_list):
            return None, target_frames_list
        
        # Case 2: List of lists of numbers
        elif all(
            isinstance(sublist, list) and all(isinstance(item, int) for item in sublist)
            for sublist in target_frames_list
        ):
            return None, target_frames_list
        
        else:
            return (
                "Error: Target Frames must be a list of numbers"
                "Valid examples: [1, 25, 45]"
            ), None
    
    except json.JSONDecodeError as e:
        return f"Error parsing Target Frames: {str(e)}", None
    except Exception as e:
        return f"Unexpected error while validating Target Frames: {str(e)}", None
    
def toggle_dataset_option(option):
    if option == "Create New Dataset":
        # Show creation container and hide selection container
        return (
            gr.update(visible=True),    # Show create_new_container
            gr.update(visible=False),   # Hide select_existing_container
            gr.update(choices=[], value=None),      # Clear Dropdown of existing datasets
            gr.update(value=""),        # Clear Dataset Name
            gr.update(value=""),         # Clear Upload Status
            gr.update(value=""),         # Clear Dataset Path
            gr.update(visible=True),     # Hide Create Dataset Button
            gr.update(visible=True),     # Show Upload Files Button 
        )
    else:
        # Hide creation container and show selection container
        datasets = get_datasets()
        return (
            gr.update(visible=False),   # Hide create_new_container
            gr.update(visible=True),    # Show select_existing_container
            gr.update(choices=datasets if datasets else [], value=None),  # Update Dropdown
            gr.update(value=""),        # Clear Dataset Name
            gr.update(value=""),         # Clear Upload Status
            gr.update(value=""),         # Clear Dataset Path
            gr.update(visible=False),     # Show Create Dataset Button
            gr.update(visible=False),     # Hide Upload Files Button
        )

# def train_model(dataset_path, config_dir, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
#                 transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
#                 gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, ar_buckets, gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print, video_clip_mode, resume_from_checkpoint, only_double_blocks
#                 ):
#     try:
#         # Validate inputs
#         if not dataset_path or not os.path.exists(dataset_path) or dataset_path == BASE_DATASET_DIR:
#             return "Error: Please provide a valid dataset path", None
        
#         os.makedirs(config_dir, exist_ok=True)

#         if not config_dir or not os.path.exists(config_dir) or config_dir == CONFIG_DIR:
#             return "Error: Please provide a valid config path", None

#         os.makedirs(output_dir, exist_ok=True)
        
#         if not output_dir or not os.path.exists(output_dir) or output_dir == OUTPUT_DIR:
#             return "Error: Please provide a valid output path", None
        
#         resolutions_error, resolutions_list = validate_resolutions(resolutions)
#         if resolutions_error:
#             return resolutions_error, None
            
#         try:
#             frame_buckets_list = json.loads(frame_buckets)
#             if not isinstance(frame_buckets_list, list) or not all(isinstance(b, int) for b in frame_buckets_list):
#                 return "Error: Frame buckets must be a list of integers. Example: [1, 33, 65]", None
#         except Exception as e:
#             return f"Error parsing frame buckets: {str(e)}", None
        
#         ar_buckets_list = None
        
#         if len(ar_buckets) > 0:
#             ar_buckets_error, ar_buckets_list = validate_ar_buckets(ar_buckets)
#             if ar_buckets_error:
#                 return ar_buckets_error, None

#         # Create configurations
#         dataset_config_path = create_dataset_config(
#             dataset_path=dataset_path,
#             config_dir=config_dir,
#             num_repeats=num_repeats,
#             resolutions=resolutions_list,
#             enable_ar_bucket=enable_ar_bucket,
#             min_ar=min_ar,
#             max_ar=max_ar,
#             num_ar_buckets=num_ar_buckets,
#             frame_buckets=frame_buckets_list,
#             ar_buckets=ar_buckets_list
#         )
        
#         training_config_path, _ = create_training_config(
#             output_dir=output_dir,
#             config_dir=config_dir,
#             dataset_config_path=dataset_config_path,
#             epochs=epochs,
#             batch_size=batch_size,
#             lr=lr,
#             save_every=save_every,
#             eval_every=eval_every,
#             rank=rank,
#             only_double_blocks=only_double_blocks,
#             dtype=dtype,
#             transformer_path=transformer_path,
#             vae_path=vae_path,
#             llm_path=llm_path,
#             clip_path=clip_path,
#             optimizer_type=optimizer_type,
#             betas=betas,
#             weight_decay=weight_decay,
#             eps=eps,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             gradient_clipping=gradient_clipping,
#             warmup_steps=warmup_steps,
#             eval_before_first_step=eval_before_first_step,
#             eval_micro_batch_size_per_gpu=eval_micro_batch_size_per_gpu,
#             eval_gradient_accumulation_steps=eval_gradient_accumulation_steps,
#             checkpoint_every_n_minutes=checkpoint_every_n_minutes,
#             activation_checkpointing=activation_checkpointing,
#             partition_method=partition_method,
#             save_dtype=save_dtype,
#             caching_batch_size=caching_batch_size,
#             steps_per_print=steps_per_print,
#             video_clip_mode=video_clip_mode
#         )

#         conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
#         conda_env_name = "pyenv"
#         num_gpus = os.getenv("NUM_GPUS", "1")
        
#         if not os.path.isfile(conda_activate_path):
#             return "Error: Conda activation script not found", None
        
#         resume_checkpoint =  "--resume_from_checkpoint" if resume_from_checkpoint else ""
        
#         cmd = (
#             f"bash -c 'source {conda_activate_path} && "
#             f"conda activate {conda_env_name} && "
#             f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus={num_gpus} "
#             f"train.py --deepspeed --config {training_config_path} {resume_checkpoint}'"          
#         )
        
#         # --regenerate_cache
            
#         proc = subprocess.Popen(
#             cmd,
#             shell=True,  # Required for complex shell commands
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             preexec_fn=os.setsid,
#             universal_newlines=False  # To handle bytes
#         )
        
#         with process_lock:
#             process_dict[proc.pid] = proc  
        
#         thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue))
#         thread.start()
        
#         pid = proc.pid
        
#         return "Training started! Logs will appear below.\n", pid

#     except Exception as e:
#         return f"Error during training: {str(e)}", None

def stop_training(pid):
    if pid is None:
        return "No training process is currently running."

    with process_lock:
        proc = process_dict.get(pid)

    if proc is None:
        return "No training process is currently running."

    if proc.poll() is not None:
        return "Training process has already finished."

    try:
        # Send SIGTERM signal to the entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)  # Wait 5 seconds to terminate
            with process_lock:
                del process_dict[pid]
            return "Training process terminated gracefully."
        except subprocess.TimeoutExpired:
            # Force termination if SIGTERM does not work
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            with process_lock:
                del process_dict[pid]
            return "Training process killed forcefully."
    except Exception as e:
        return f"Error stopping training process: {str(e)}"

def upload_dataset(files, current_dataset, action, dataset_name=None):
    """
    Handle uploaded dataset files and store them in a unique directory.
    Action can be 'start' (initialize a new dataset) or 'add' (add files to current dataset).
    """
    if action == "start":
        if not dataset_name:
            return current_dataset, "Please provide a dataset name.", []
        # Ensure the dataset name does not contain invalid characters
        dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        dataset_dir = os.path.join(DATASET_DIR, dataset_name)
        if os.path.exists(dataset_dir):
            return current_dataset, f"Dataset '{dataset_name}' already exists. Please choose a different name.", []
        
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_dir_images = os.path.join(dataset_dir, "images")
        dataset_dir_videos = os.path.join(dataset_dir, "videos")
        os.makedirs(dataset_dir_images, exist_ok=True)
        os.makedirs(dataset_dir_videos, exist_ok=True)
        
        return dataset_dir, f"Started new dataset: {dataset_dir}", show_media(dataset_dir)
    
    if not current_dataset:
        return current_dataset, "Please start a new dataset before uploading files.", []
    
    if not files:
        return current_dataset, "No files uploaded.", []
    
    # Calculate the total size of the current dataset
    total_size = 0
    for root, dirs, files_in_dir in os.walk(current_dataset):
        for f in files_in_dir:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)

    # Calculate the size of the new files
    new_files_size = 0
    for file in files:
        if IS_CONTAINER:
            new_files_size += os.path.getsize(file.name)

    # Check if adding these files would exceed the limit
    if IS_CONTAINER and (total_size + new_files_size) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return current_dataset, f"Upload would exceed the {MAX_UPLOAD_SIZE_MB}MB limit. Please upload smaller files or finalize the dataset.", show_media(current_dataset)

    uploaded_files = []
    unsupported_files = []
    
    # Temporary storage for media and captions
    media_files = []      # List of tuples: (file_object, destination_path)
    caption_files = []    # List of tuples: (file_object, original_name)
    extracted_files = []  # List of extracted file paths (for further processing)

    for file in files:
        file_path = file.name
        filename = os.path.basename(file_path)

        if zipfile.is_zipfile(file_path):
            # If the file is a ZIP, extract its contents
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract to a temporary directory within the current_dataset
                    temp_extract_dir = os.path.join(current_dataset, "__temp_extract__")
                    zip_ref.extractall(temp_extract_dir)
                uploaded_files.append(f"{filename} (extracted)")
                # Collect all extracted file paths for processing
                for root, dirs, extracted in os.walk(temp_extract_dir):
                    for extracted_file in extracted:
                        extracted_file_path = os.path.join(root, extracted_file)
                        extracted_files.append(extracted_file_path)
            except zipfile.BadZipFile:
                uploaded_files.append(f"{filename} (invalid ZIP)")
                continue
        else:
            # Categorize files based on their extensions
            lower_filename = filename.lower()
            if lower_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4')):
                # Determine the destination directory
                if lower_filename.endswith('.mp4'):
                    dest_dir = os.path.join(current_dataset, "videos")
                else:
                    dest_dir = os.path.join(current_dataset, "images")
                
                dest_path = os.path.join(dest_dir, filename)
                media_files.append((file, dest_path))
                uploaded_files.append(filename)
            elif lower_filename.endswith('.txt'):
                # Store caption files for later processing
                caption_files.append((file, filename))
            else:
                unsupported_files.append(filename)
    
    # Process extracted files from ZIP archives
    for extracted_file_path in extracted_files:
        extracted_filename = os.path.basename(extracted_file_path)
        lower_filename = extracted_filename.lower()

        if lower_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4')):
            if lower_filename.endswith('.mp4'):
                dest_dir = os.path.join(current_dataset, "videos")
            else:
                dest_dir = os.path.join(current_dataset, "images")
            
            dest_path = os.path.join(dest_dir, extracted_filename)
            try:
                shutil.move(extracted_file_path, dest_path)
                uploaded_files.append(extracted_filename)
            except Exception as e:
                uploaded_files.append(f"{extracted_filename} (failed to upload: {e})")
        elif lower_filename.endswith('.txt'):
            caption_files.append((extracted_file_path, extracted_filename))
        else:
            # Unsupported file inside ZIP
            uploaded_files.append(f"{extracted_filename} (unsupported format)")
    
    # Remove temporary extraction directory
    temp_extract_dir = os.path.join(current_dataset, "__temp_extract__")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    
    # Check upload size constraints for media files
    if IS_CONTAINER:
        new_media_size = 0
        for _, dest_path in media_files:
            new_media_size += os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if (total_size + new_media_size) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            return current_dataset, f"Upload would exceed the {MAX_UPLOAD_SIZE_MB}MB limit. Please upload smaller files or finalize the dataset.", show_media(current_dataset)
    
    # Upload media files
    for file_obj, dest_path in media_files:
        try:
            shutil.copy(file_obj.name, dest_path)
        except Exception as e:
            uploaded_files.append(f"{os.path.basename(dest_path)} (failed to upload: {e})")
    
    # Upload caption files
    for caption in caption_files:
        if isinstance(caption[0], str):
            # Extracted caption file from ZIP
            caption_file_path, caption_filename = caption
            source_path = caption_file_path
        else:
            # Uploaded caption file
            file_obj, caption_filename = caption
            source_path = file_obj.name
        
        base_name, _ = os.path.splitext(caption_filename)
        # Search for a media file with the same base name in images and videos
        matched_media_dir = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4']:
            for media_dir in ["images", "videos"]:
                media_path = os.path.join(current_dataset, media_dir, base_name + ext)
                if os.path.exists(media_path):
                    matched_media_dir = media_dir
                    break
            if matched_media_dir:
                break
        
        if matched_media_dir:
            dest_path = os.path.join(current_dataset, matched_media_dir, caption_filename)
            try:
                shutil.copy(source_path, dest_path)
                uploaded_files.append(f"{caption_filename} (caption for {base_name})")
            except Exception as e:
                uploaded_files.append(f"{caption_filename} (failed to upload: {e})")
        else:
            # No matching media file found; handle as needed
            uploaded_files.append(f"{caption_filename} (no matching media file found)")
        
        # If the caption file was extracted from ZIP, remove it after processing
        if isinstance(caption[0], str):
            try:
                os.remove(caption_file_path)
            except Exception:
                pass
    
    # Handle unsupported files
    for unsupported in unsupported_files:
        uploaded_files.append(f"{unsupported} (unsupported format)")
    
    return current_dataset, f"Uploaded files: {', '.join(uploaded_files)}", show_media(current_dataset)


def update_ui_with_config(config_values):
    """
    Updates Gradio interface components with configuration values.

    Args:
        config_values (dict): Dictionary containing dataset and training configurations.

    Returns:
        tuple: Updated values for the interface components.
    """
    # Define default values for each field
    defaults = {
        "epochs": 1000,
        "batch_size": 1,
        "lr": 2e-5,
        "save_every": 2,
        "eval_every": 1,
        "rank": 32,
        "only_double_blocks": False,
        "dtype": "bfloat16",
        "transformer_path": "",
        "vae_path": "",
        "llm_path": "",
        "clip_path": "",
        "optimizer_type": "adamw_optimi",
        "betas": json.dumps([0.9, 0.99]),
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_accumulation_steps": 4,
        "num_repeats": 10,
        "resolutions_input": json.dumps([512]),
        "enable_ar_bucket": True,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 7,
        "ar_buckets": None,
        "frame_buckets": json.dumps([1, 33, 65]),
        "gradient_clipping": 1.0,
        "warmup_steps": 100,
        "eval_before_first_step": True,
        "eval_micro_batch_size_per_gpu": 1,
        "eval_gradient_accumulation_steps": 1,
        "checkpoint_every_n_minutes": 120,
        "activation_checkpointing": True,
        "partition_method": "parameters",
        "save_dtype": "bfloat16",
        "caching_batch_size": 1,
        "steps_per_print": 1,
        "video_clip_mode": "single_middle"
    }

    # Helper function to get values with defaults
    def get_value(key):
        return config_values.get(key, defaults.get(key))

    # Extract values with error handling
    try:
        epochs = get_value("epochs")
        batch_size = get_value("batch_size")
        lr = get_value("lr")
        save_every = get_value("save_every")
        eval_every = get_value("eval_every")
        rank = get_value("rank")
        only_double_blocks = get_value("only_double_blocks")
        dtype = get_value("dtype")
        transformer_path = get_value("transformer_path")
        vae_path = get_value("vae_path")
        llm_path = get_value("llm_path")
        clip_path = get_value("clip_path")
        optimizer_type = get_value("optimizer_type")
        betas = get_value("betas")
        weight_decay = get_value("weight_decay")
        eps = get_value("eps")
        gradient_accumulation_steps = get_value("gradient_accumulation_steps")
        num_repeats = get_value("num_repeats")
        resolutions_input = get_value("resolutions_input")
        enable_ar_bucket = get_value("enable_ar_bucket")
        min_ar = get_value("min_ar")
        max_ar = get_value("max_ar")
        num_ar_buckets = get_value("num_ar_buckets")
        ar_buckets = get_value("ar_buckets")
        frame_buckets = get_value("frame_buckets")
        gradient_clipping = get_value("gradient_clipping")
        warmup_steps = get_value("warmup_steps")
        eval_before_first_step = get_value("eval_before_first_step")
        eval_micro_batch_size_per_gpu = get_value("eval_micro_batch_size_per_gpu")
        eval_gradient_accumulation_steps = get_value("eval_gradient_accumulation_steps")
        checkpoint_every_n_minutes = get_value("checkpoint_every_n_minutes")
        activation_checkpointing = get_value("activation_checkpointing")
        partition_method = get_value("partition_method")
        save_dtype = get_value("save_dtype")
        caching_batch_size = get_value("caching_batch_size")
        steps_per_print = get_value("steps_per_print")
        video_clip_mode = get_value("video_clip_mode")
    except Exception as e:
        print(f"Error extracting configurations: {str(e)}")
        # Return default values in case of an error
        epochs = defaults["epochs"]
        batch_size = defaults["batch_size"]
        lr = defaults["lr"]
        save_every = defaults["save_every"]
        eval_every = defaults["eval_every"]
        rank = defaults["rank"]
        only_double_blocks = defaults["only_double_blocks"]
        dtype = defaults["dtype"]
        transformer_path = defaults["transformer_path"]
        vae_path = defaults["vae_path"]
        llm_path = defaults["llm_path"]
        clip_path = defaults["clip_path"]
        optimizer_type = defaults["optimizer_type"]
        betas = defaults["betas"]
        weight_decay = defaults["weight_decay"]
        eps = defaults["eps"]
        gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
        num_repeats = defaults["num_repeats"]
        resolutions_input = defaults["resolutions_input"]
        enable_ar_bucket = defaults["enable_ar_bucket"]
        min_ar = defaults["min_ar"]
        max_ar = defaults["max_ar"]
        num_ar_buckets = defaults["num_ar_buckets"]
        ar_buckets = defaults["ar_buckets"]
        frame_buckets = defaults["frame_buckets"]
        gradient_clipping = defaults["gradient_clipping"]
        warmup_steps = defaults["warmup_steps"]
        eval_before_first_step = defaults["eval_before_first_step"]
        eval_micro_batch_size_per_gpu = defaults["eval_micro_batch_size_per_gpu"]
        eval_gradient_accumulation_steps = defaults["eval_gradient_accumulation_steps"]
        checkpoint_every_n_minutes = defaults["checkpoint_every_n_minutes"]
        activation_checkpointing = defaults["activation_checkpointing"]
        partition_method = defaults["partition_method"]
        save_dtype = defaults["save_dtype"]
        caching_batch_size = defaults["caching_batch_size"]
        steps_per_print = defaults["steps_per_print"]
        video_clip_mode = defaults["video_clip_mode"]
    print(num_repeats)
    return (
        epochs,
        batch_size,
        lr,
        save_every,
        eval_every,
        rank,
        only_double_blocks,
        dtype,
        transformer_path,
        vae_path,
        llm_path,
        clip_path,
        optimizer_type,
        betas,
        weight_decay,
        eps,
        gradient_accumulation_steps,
        num_repeats,
        resolutions_input,
        enable_ar_bucket,
        min_ar,
        max_ar,
        num_ar_buckets,
        ar_buckets,
        frame_buckets,
        gradient_clipping,
        warmup_steps,
        eval_before_first_step,
        eval_micro_batch_size_per_gpu,
        eval_gradient_accumulation_steps,
        checkpoint_every_n_minutes,
        activation_checkpointing,
        partition_method,
        save_dtype,
        caching_batch_size,
        steps_per_print,
        video_clip_mode
    )

def show_media(dataset_dir):
    """Display uploaded images and .mp4 videos in a single gallery."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        # Return an empty list if the dataset_dir is invalid
        return []
    
    dataset_images = os.path.join(dataset_dir, "images")
    dataset_videos = os.path.join(dataset_dir, "videos")
    
    # Supported file extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp4')
    
    # Initialize lists for images and videos
    image_files = []
    video_files = []
    
    # Check if the images directory exists and list image files
    if os.path.isdir(dataset_images):
        image_files = [
            os.path.join(dataset_images, f)
            for f in os.listdir(dataset_images)
            if f.lower().endswith(valid_extensions)
        ]
    
    # Check if the videos directory exists and list video files
    if os.path.isdir(dataset_videos):
        video_files = [
            os.path.join(dataset_videos, f)
            for f in os.listdir(dataset_videos)
            if f.lower().endswith(valid_extensions)
        ]

    # Combine image and video file paths
    media_paths = image_files + video_files
    
    # Limit the number of media files if necessary
    media_paths = media_paths[:MAX_MEDIA]
    
    # Filter out any paths that do not exist
    existing_media = [f for f in media_paths if os.path.exists(f)]
    
    return existing_media

def create_zip(dataset_name, download_dataset, download_config, download_outputs):
    """
    Creates a ZIP archive containing the dataset, config, and output directories.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: Path to the created ZIP file.
    """
    try:
        # Define paths
        dataset_dir = os.path.join(DATASET_DIR, dataset_name)
        config_dir_path = os.path.join(CONFIG_DIR, dataset_name)
        output_dir_path = os.path.join(OUTPUT_DIR, dataset_name)

        # Check if all directories exist
        # if not all([os.path.exists(dataset_dir), os.path.exists(config_dir_path), os.path.exists(output_dir_path)]):
        #     return None, "One or more directories (dataset, config, output) do not exist."

        # Create a temporary directory to store the ZIP
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{dataset_name}_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add dataset directory
            
            if download_dataset:
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=os.path.dirname(dataset_dir))
                        zipf.write(file_path, arcname)

            if download_config:
                # Add config directory
                for root, dirs, files in os.walk(config_dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=os.path.dirname(config_dir_path))
                        zipf.write(file_path, arcname)

            if download_outputs:
                # Add output directory
                for root, dirs, files in os.walk(output_dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=os.path.dirname(output_dir_path))
                        zipf.write(file_path, arcname)
            
                
            
                

        return zip_path, None

    except Exception as e:
        return None, f"Error creating ZIP archive: {str(e)}"

def handle_download(dataset_path, selected_files):
    """
    Handles the download button click by creating a ZIP and returning it for download.
    
    Args:
        dataset_path (str): Path to the dataset.
    
    Returns:
        tuple: File object for download and status message.
    """
    
    try:
        if not dataset_path or dataset_path == DATASET_DIR or not os.path.exists(dataset_path):
            return None, "Invalid dataset path."

        dataset_name = os.path.basename(dataset_path)
        
        download_dataset = "Dataset" in selected_files
        download_config = "Configs" in selected_files
        download_outputs = "Outputs" in selected_files
        
        zip_path, error = create_zip(dataset_name,  download_dataset, download_config, download_outputs)

        if error:
            return None, error

        # Return the ZIP file for download
        return zip_path, "Download ready."
    
    except Exception as e:
        return None, f"Error during download: {str(e)}"

def cleanup_temp_files(temp_dir, retention_time=3600):
    """
    Periodically cleans up temporary directories older than retention_time seconds.
    
    Args:
        temp_dir (str): Path to the temporary directory.
        retention_time (int): Time in seconds to retain the files.
    """
    while True:
        try:
            now = time.time()
            for folder in os.listdir(temp_dir):
                folder_path = os.path.join(temp_dir, folder)
                if os.path.isdir(folder_path):
                    folder_mtime = os.path.getmtime(folder_path)
                    if now - folder_mtime > retention_time:
                        shutil.rmtree(folder_path)
            time.sleep(1800)  # Check every 30 minutes
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            time.sleep(1800)

# Start the cleanup thread
temp_base_dir = tempfile.mkdtemp()
cleanup_thread = threading.Thread(target=cleanup_temp_files, args=(temp_base_dir,), daemon=True)
cleanup_thread.start()


def train(dit_path, 
            vae_path,
            llm_path,
            clip_path,
            dataset_path,
            config_path, 
            output_path, 
            dataset_name,
            mixed_precision,
            optimizer_type,
            learning_rate,
            gradient_checkpointing,
            gradient_accumulation_steps,
            max_data_loader_n_workers,
            persistent_data_loader_workers,
            network_dim,
            max_train_epochs,
            save_every_n_epochs,
            seed,
            fp8_base,
            enable_lowvram,
            blocks_to_swap,
            attention,
            general_batch_size,
            general_resolutions,
            general_enable_bucket,
            general_bucket_no_upscale,
            image_resolutions,
            image_batch_size,
            image_enable_bucket,
            image_bucket_no_upscale,
            video_resolutions,
            video_batch_size,
            target_frames,
            frame_extraction,
            frame_stride,
            frame_sample,
            video_enable_bucket,
            video_bucket_no_upscale):
        try:
            os.makedirs(config_path, exist_ok=True)
        
            if not config_path or not os.path.exists(config_path) or config_path == CONFIG_DIR:
                return "Error: Please provide a valid config path", None

            os.makedirs(output_path, exist_ok=True)
            
            if not output_path or not os.path.exists(output_path) or output_path == OUTPUT_DIR:
                return "Error: Please provide a valid output path", None
            
            general_resolutions_error, general_resolutions_list = validate_resolutions(general_resolutions)
            if general_resolutions_error:
                return general_resolutions_error, None
            
            general_config = GeneralConfig(general_resolutions_list, caption_extension=".txt", batch_size=general_batch_size, enable_bucket=general_enable_bucket, bucket_no_upscale=general_bucket_no_upscale)
            
            dataset_path = os.path.join(DATASET_DIR, dataset_name)
            dataset_folder = Path(dataset_path)
            
            dataset_images_path = os.path.join(dataset_path, "images")
            dataset_videos_path = os.path.join(dataset_path, "videos")
            
            dataset_images_folder = Path(dataset_images_path)
            dataset_videos_folder = Path(dataset_videos_path)
            
            target_frames_error, target_frames_list = validate_target_frames(target_frames)
            if target_frames_error:
                return target_frames_error, None
            
            if not dataset_folder.is_dir():
                print(f"The path {dataset_path} is not a valid directory.")
                return
            
            dataset_images = list(dataset_images_folder.iterdir())
            dataset_videos = list(dataset_videos_folder.iterdir())
            
            has_images = False
            has_videos = False
            
            has_images = len(dataset_images) > 0
            has_videos = len(dataset_videos) > 0
            
            datasets_configs = []
            
            if has_images:
                image_resolutions_error, image_resolutions_list = validate_resolutions(image_resolutions)
                if image_resolutions_error:
                    return image_resolutions_error, None
                
                if (not general_resolutions_list or len(general_resolutions_list) != 2) and (not image_resolutions_list or len(image_resolutions_list) != 2):
                    return "Error: Please provide general resolutions or image resolutions in the format: [Width, Height]", None
                
                datasets_configs.append(ImageDataset(image_directory=dataset_images_path, resolution=image_resolutions_list, batch_size=image_batch_size, enable_bucket=image_enable_bucket, bucket_no_upscale=image_bucket_no_upscale))
            
            if has_videos:
                video_resolutions_error, video_resolutions_list = validate_resolutions(video_resolutions)
                if video_resolutions_error:
                    return video_resolutions_error, None
                
                if (not general_resolutions_list or len(general_resolutions_list) != 2) and (not video_resolutions_list or len(video_resolutions_list) != 2):
                    return "Error: Please provide general resolutions or video resolutions in the format: [Width, Height]", None
                
                datasets_configs.append(VideoDataset(video_directory=dataset_videos_path, resolution=video_resolutions_list, batch_size=video_batch_size, target_frames=target_frames_list, frame_extraction=frame_extraction, frame_stride=frame_stride, frame_sample=frame_sample, enable_bucket=video_enable_bucket, bucket_no_upscale=video_bucket_no_upscale))
                
            config = DatasetConfig(general_config, datasets_configs)
            
            config.save(config_path)
            
            training_config = TrainingConfig(
                dit_path=dit_path,
                vae_path=vae_path,
                llm_path=llm_path,
                clip_path=clip_path,
                dataset_config=f"{config_path}/dataset_config.toml",
                output_dir=output_path,
                output_name=dataset_name,
                log_dir=f"{output_path}/logs",
                mixed_precision=mixed_precision,
                optimizer_type=optimizer_type,
                learning_rate=learning_rate,
                gradient_checkpointing=gradient_checkpointing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_data_loader_n_workers=max_data_loader_n_workers,
                persistent_data_loader_workers=persistent_data_loader_workers,
                network_module="lora.networks",
                network_dim=network_dim,
                network_alpha=network_alpha,
                max_train_epochs=max_train_epochs,
                save_every_n_epochs=save_every_n_epochs,
                seed=seed,
                num_cpu_threads_per_process=1,
                fp8_base=fp8_base,
                enable_lowvram=enable_lowvram,
                blocks_to_swap=blocks_to_swap,
                attention=attention
            )
        
            training_config.save(config_path)
            
            conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
            conda_env_name = "pyenv"
            
            if not os.path.isfile(conda_activate_path):
                return "Error: Conda activation script not found", None
            
            cmd = (
                f"bash -c 'source {conda_activate_path} && "
                f"conda activate {conda_env_name} && "
                f"{training_config.generate_command()}'"
            )
            
            proc = subprocess.Popen(
                cmd,
                shell=True,  # Required for complex shell commands
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=False  # To handle bytes
            )
        
            with process_lock:
                process_dict[proc.pid] = proc
                
            thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue))
            thread.start()
        
            pid = proc.pid
            
            return "Training started! Logs will appear below.\n", pid
            
        except Exception as e:
            return f"Error during training: {str(e)}", None
        
theme = gr.themes.Monochrome(
    primary_hue="gray",
    secondary_hue="gray",
    neutral_hue="gray",
    text_size=gr.themes.Size(
        lg="18px", 
        md="15px", 
        sm="13px", 
        xl="22px", 
        xs="12px", 
        xxl="24px", 
        xxs="9px"
    ),
    font=[
        gr.themes.GoogleFont("Source Sans Pro"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif"
    ]
)

# Gradio Interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# LoRA Training Interface for Hunyuan Video")
    
    gr.Markdown("### Step 1: Dataset Management\nChoose to create a new dataset or select an existing one.")
    
    with gr.Row():
        dataset_option = gr.Radio(
            choices=["Create New Dataset", "Select Existing Dataset"],
            value="Create New Dataset",
            label="Dataset Option"
        )
    
    def handle_start_dataset(dataset_name):
        if not dataset_name.strip():
            return (
                gr.update(value="Please provide a dataset name."), 
                gr.update(value=None), 
                gr.update(visible=True),   # Keep button visible
                gr.update(visible=False),
                gr.update(value=""),
                # gr.update(visible=False)  # Hide download button
            )
        dataset_path, message, media = upload_dataset([], None, "start", dataset_name=dataset_name)
        if "already exists" in message:
            return (
                gr.update(value=message), 
                gr.update(value=None), 
                gr.update(visible=True),   # Keep button visible
                gr.update(visible=False),
                gr.update(value=""),
                # gr.update(visible=False)  # Hide download button
            )
        return (
            gr.update(value=message), 
            dataset_path, 
            gr.update(visible=False), 
            gr.update(visible=True),
            gr.update(value=dataset_path),
            # gr.update(visible=True)    # Show download button
        )
    

    with gr.Row(visible=True, elem_id="create_new_dataset_container") as create_new_container:
            with gr.Column():
                with gr.Row():
                    dataset_name_input = gr.Textbox(
                        label="Dataset Name",
                        placeholder="Enter your dataset name",
                        interactive=True
                    )
                create_dataset_button = gr.Button("Create Dataset", interactive=False)  # Initially disabled
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                upload_files = gr.File(
                    label="Upload Images (.jpg, .png, .gif, .bmp, .webp), Videos (.mp4), Captions (.txt) or a ZIP archive",
                    file_types=[".jpg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".txt", ".zip"],
                    file_count="multiple",
                    type="filepath", 
                    interactive=True,
                    visible=False
                )
                
    # Function to enable/disable the "Start New Dataset" button based on input
    def toggle_start_button(name):
        if name.strip():
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)
    
    current_dataset_state = gr.State(None)
    training_process_pid = gr.State(None)
    
    dataset_name_input.change(
        fn=toggle_start_button, 
        inputs=dataset_name_input, 
        outputs=create_dataset_button
    )
    
    def handle_upload(files, current_dataset):
        updated_dataset, message, media = upload_dataset(files, current_dataset, "add")
        return updated_dataset, message, media
    
    # Container to select existing dataset
    with gr.Row(visible=False, elem_id="select_existing_dataset_container") as select_existing_container:
        with gr.Column():
            existing_datasets = gr.Dropdown(
                choices=[],  # Initially empty; will be updated dynamically
                label="Select Existing Dataset",
                interactive=True
            )
    
    # 2. Media Gallery
    gr.Markdown("### Dataset Preview")
    gallery = gr.Gallery(
        label="Dataset Preview",
        show_label=False,
        elem_id="gallery",
        columns=3,
        rows=2,
        object_fit="contain",
        height="auto",
        visible=True
    )
    
   
    
    # Upload files and update gallery
    upload_files.upload(
        fn=lambda files, current_dataset: handle_upload(files, current_dataset),
        inputs=[upload_files, current_dataset_state],
        outputs=[current_dataset_state, upload_status, gallery],
        queue=True
    )
    
    # Function to handle selecting an existing dataset and updating the gallery
    def handle_select_existing(selected_dataset):
        if selected_dataset:
            dataset_path = os.path.join(DATASET_DIR, selected_dataset)
            config, error = load_training_config(selected_dataset)
            if error:
                return (
                    "",  # Clear dataset path
                    "",  # Clear config and output paths
                    "",  # Clear parameter values
                    f"Error loading configuration: {error}",
                    [],    # Clear gallery
                    # gr.update(visible=False),    # Hide download button
                    gr.update(value=""),         # Clear download status
                    {}
                )
            config_values = extract_config_values(config)
            
            # Update config and output paths
            config_path = os.path.join(CONFIG_DIR, selected_dataset)
            output_path = os.path.join(OUTPUT_DIR, selected_dataset)
            
            return (
                dataset_path,  # Update dataset_path
                config_path,   # Update config_dir
                output_path,   # Update output_dir
                "",            # Clear error messages
                show_media(dataset_path),  # Update gallery with dataset files
                # gr.update(visible=True),    # Show download button
                gr.update(value=""),        # Clear download status
                config_values  # Update training parameters
            )
        return "", "", "", "No dataset selected.", [], gr.update(value=""), {} #, gr.update(visible=False), 

    with gr.Row():
        with gr.Column():
            dataset_path = gr.Textbox(
                label="Dataset Path",
                value=DATASET_DIR,
                interactive=False
            )
            config_dir = gr.Textbox(
                label="Config Path",
                value=CONFIG_DIR,
                interactive=False
            )
            output_dir = gr.Textbox(
                label="Output Path",
                value=OUTPUT_DIR,
                interactive=False
            )
    
    create_dataset_button.click(
        fn=handle_start_dataset,
        inputs=dataset_name_input,
        outputs=[upload_status, current_dataset_state, create_dataset_button, upload_files, dataset_path] #, download_button, download_zip]
    )
   
    dataset_option.change(
        fn=toggle_dataset_option,
        inputs=dataset_option,
        outputs=[create_new_container, select_existing_container, existing_datasets, dataset_name_input, upload_status, dataset_path, create_dataset_button, upload_files] #, download_button]
    )
    
    # Update config path and output path
    def update_config_output_path(dataset_path):
        config_path = os.path.join(CONFIG_DIR, os.path.basename(dataset_path))
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(dataset_path))
        return config_path, output_path
    
     
    # Update gallery when dataset path changes
    dataset_path.change(
        fn=lambda path: show_media(path),
        inputs=dataset_path,
        outputs=gallery
    )
    
    dataset_path.change(
        fn=update_config_output_path,
        inputs=dataset_path,
        outputs=[config_dir, output_dir]
    )
    
    # Handle Models Configurations
    gr.Markdown("#### Models Configurations")
    with gr.Row():
        with gr.Column():
            dit_path = gr.Textbox(
                label="DIT Path",
                value=os.path.join(MODEL_DIR, "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"),
                info="Path to the DIT model weights for Hunyuan Video."
            )
            vae_path = gr.Textbox(
                label="VAE Path",
                value=os.path.join(MODEL_DIR, "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
                info="Path to the VAE model file."
            )
            llm_path = gr.Textbox(
                label="LLM Path",
                value=os.path.join(MODEL_DIR, "text_encoder"),
                info="Path to the LLM's text tokenizer and encoder."
            )
            clip_path = gr.Textbox(
                label="CLIP Path",
                value=os.path.join(MODEL_DIR, "text_encoder_2"),
                info="Path to the CLIP model directory."
            )
            
    gr.Markdown("### Step 2: Training\nConfigure your training parameters and start or stop the training process.")
    with gr.Column():
        gr.Markdown("#### Training Parameters")
        with gr.Row():
            max_train_epochs = gr.Number(
                label="Max Training Epochs",
                value=16,
                info="Max number of training epochs",
            )
            general_batch_size = gr.Number(
                label="General Batch Size",
                value=1,
                info="This is the default batch size for all datasets"
            )
            lr = gr.Number(
                label="Learning Rate",
                value=1e-3,
                step=0.001,
                info="Optimizer learning rate"
            )
            gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                value=1,
                info="Gradient Accumulation Steps"
            )
            save_every = gr.Number(
                label="Save Every N Epochs",
                value=2,
                info="Frequency to save checkpoints"
            )
        with gr.Row():
            network_dim = gr.Number(
                label="Network dim",
                value=32,
                precision=0,
                info="Network dimension (2-128)"
            )
            
            network_alpha = gr.Number(
                label="Network alpha",
                value=16,
                precision=0,
                info="Network alpha (2-128)"
            )
            
            mixed_precision_data_type = gr.Dropdown(
                label="Mixed Precision Data Type",
                choices=['fp32', 'fp16', 'bf16', 'fp8'],
                value="bf16",
            )
        
        # Dataset configuration fields
        gr.Markdown("#### Dataset Configuration")
        with gr.Row():
            general_resolutions = gr.Textbox(
                label="General Resolutions",
                value="",
                info="[W, H], default is None. This is the default resolution for all datasets. Example: [512, 512]"
            )
             
            general_enable_bucket = gr.Checkbox(
                label="General Enable Bucket",
                value=False,
                info="Enable bucketing for datasets"
            )
            
            general_bucket_no_upscale = gr.Checkbox(
                label="General Bucket NO Upscale",
                value=False,
                info="Disable upscaling for bucketing. Ignored if enable_bucket is false"
            )
        
        gr.Markdown("#### Image Dataset Configuration")
        with gr.Row():
            image_resolutions = gr.Textbox(
                label="Image Resolutions",
                value=None,
                info="(required) if general resolution is not set. [W, H] default is None"
            )
            
            image_batch_size = gr.Number(
                label="Image Batch Size",
                value=lambda: None,
                info="(optional) will overwrite the default general resolutions setting"
            )
            
            image_enable_bucket = gr.Checkbox(
                label="Image Enable Bucket",
                value=None,
                info="(optional) will overwrite the default general bucketing setting"
            )
            
            image_bucket_no_upscale = gr.Checkbox(
                label="Image Bucket NO Upscale",
                value=None,
                info="(optional) will overwrite the default general bucketing setting"
            )
            
        gr.Markdown("#### Video Dataset Configuration")
        with gr.Row():
            video_resolutions = gr.Textbox(
                label="Video Resolutions",
                value=None,
                info="(required) if general resolution is not set. [W, H] default is None"
            )
            
            video_batch_size = gr.Number(
                label="Video Batch Size",
                value=lambda: None,
                info="(optional) will overwrite the default general resolutions setting"
            )
            
            target_frames = gr.Textbox(
                label="Target Frames",
                value="[1, 25, 45]",
                info="Required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)"
            )
            
        with gr.Row():
            frame_extraction = gr.Dropdown(
                label="Frame Extraction",
                choices=['head', 'chunk', 'slide', 'uniform'],
                value="head",
                info="Method to extract frames from videos"
            )
            
            frame_stride = gr.Number(
                label="Frame Stride",
                value=1,
                info="Available for 'slide' frame extraction"
            )
            
            frame_sample = gr.Number(
                label="Frame Stride",
                value=1,
                info="Available for 'uniform' frame extraction"
            )
            
        with gr.Row():
            video_enable_bucket = gr.Checkbox(
                label="Video Enable Bucket",
                value=None,
                info="(optional) will overwrite the default general bucketing setting"
            )
            video_bucket_no_upscale = gr.Checkbox(
                label="Video Bucket NO Upscale",
                value=None,
                info="(optional) will overwrite the default general bucketing setting"
            )
           
            
        gr.Markdown("#### Optimizer Parameters")
        with gr.Row():
            optimizer_type = gr.Dropdown(
                label="Optimizer Type",
                choices=['adamw', 'adamw8bit', 'adamw_optimi', 'stableadamw', 'sgd', 'adamw8bitKahan'],
                value="adamw8bit",
                info="Type of optimizer"
            )
            
            max_data_loader_n_workers = gr.Number(
                label="Max DataLoader N Workers",
                value=2,
                info="Max number of workers for data loader"
            )
            
        # Additional training parameters
        gr.Markdown("#### Additional Training Parameters")
        with gr.Row():
            attention = gr.Dropdown(
                label="Attention",
                choices=['sdpa', 'sage_attn', 'flash_attn', 'xformers'],
                value="sdpa",
                info=""
            )
            fp8_base = gr.Checkbox(
                label="FP8 Base",
                value=True,
                info="Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality"
            )
            enable_lowvram = gr.Checkbox(
                label="Enable Low VRAM",
                value=False,
                info="VRAM: 12GB or more recommended for image training, 24GB or more recommended for video training (For 12GB, use a resolution of 960x544 or lower and use memory-saving enable this option)"
            )
            
            blocks_to_swap = gr.Number(
                label="Blocks to Swap",
                value=20,
                precision=0,
                visible=False,
                info="Number of blocks to swap (20-36)"
            )
            
            def toggle_blocks_swap(checked):
                return gr.update(visible=checked)

            enable_lowvram.change(
                toggle_blocks_swap,
                inputs=enable_lowvram,
                outputs=blocks_to_swap
            )
        
            # block_swap = gr.Checkbox(
            #     label="FP8 Base",
            #     value=True,
            #     info="Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality"
            # )
            gradient_checkpointing = gr.Checkbox(
                label="Gradiente Checkpointing",
                value=True,
                info="Enable Gradient Checkpointing"
            )
            persistent_data_loader_workers = gr.Checkbox(
                label="Persistent Data Loader Workers",
                value=True,
                info="Enable Persistent Data Loader Workers"
            )
            seed = gr.Number(
                label="Seed",
                value=42,
                info=""
            )
       
        with gr.Row():
            with gr.Column(scale=1):
                resume_from_checkpoint = gr.Checkbox(label="Resume from last checkpoint", info="If this is your first training, do not check this box, because the output folder will not have a checkpoint (global_step....) and will cause an error")
                
                only_double_blocks = gr.Checkbox(label="Train only double blocks (Experimental)", info="This option will be used to train only double blocks, some people report that training only double blocks can reduce the amount of motion blur and improve the final quality of the video.")
                
                train_button = gr.Button("Start Training", visible=True)
                stop_button = gr.Button("Stop Training", visible=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        output = gr.Textbox(
                            label="Output Logs",
                            lines=20,
                            interactive=False,
                            elem_id="log_box"
                        )
                        
    hidden_config = gr.JSON(label="Hidden Configuration", visible=False)
    
     # Adding Download Section
    gr.Markdown("### Download Files")
    
    with gr.Row():
        explorer = gr.FileExplorer(root_dir="/workspace", interactive=False, label="File Explorer")
        
        
    with gr.Row():
        with gr.Column():
            download_options = gr.CheckboxGroup(["Outputs", "Dataset", "Configs"], label="Download Options"),
            download_button = gr.Button("Download ZIP", visible=True)
        download_zip = gr.File(label="Download ZIP", visible=True)
        download_status = gr.Textbox(label="Download Status", interactive=False, visible=True)
        
    
    
    def handle_train_click(dit_path,
                           vae_path,
                           llm_path,
                           clip_path,
                           dataset_path, 
                           config_path,
                           output_path, 
                           dataset_name,
                           mixed_precision,
                           optimizer_type,
                           learning_rate,
                           gradient_checkpointing,
                           gradient_accumulation_steps,
                           max_data_loader_n_workers,
                           persistent_data_loader_workers,
                           network_dim,
                           network_alpha,
                           max_train_epochs,
                           save_every_n_epochs,
                           seed,
                           fp8_base,
                           enable_lowvram,
                           blocks_to_swap,
                           attention,
                           general_batch_size,
                           general_resolutions,
                           general_enable_bucket,
                           general_bucket_no_upscale,
                           image_resolutions,
                           image_batch_size,
                           image_enable_bucket,
                           image_bucket_no_upscale,
                           video_resolutions,
                           video_batch_size,
                           target_frames,
                           frame_extraction,
                           frame_stride,
                           frame_sample,
                           video_enable_bucket,
                           video_bucket_no_upscale,
                           ):
        
        with process_lock:
            if process_dict:
                return "A training process is already running. Please stop it before starting a new one.", training_process_pid, gr.update(interactive=False)
            
        if image_enable_bucket == False:
            image_enable_bucket = None
            
        if image_bucket_no_upscale == False:
            image_bucket_no_upscale = None
            
        if video_enable_bucket == False:
            video_enable_bucket = None
        
        if video_bucket_no_upscale == False:
            video_bucket_no_upscale = None
            
        
        message, pid = train(
            dit_path, 
            vae_path,
            llm_path,
            clip_path,
            dataset_path, 
            config_path,
            output_path, 
            dataset_name,
            mixed_precision,
            optimizer_type,
            learning_rate,
            gradient_checkpointing,
            gradient_accumulation_steps,
            max_data_loader_n_workers,
            persistent_data_loader_workers,
            network_dim,
            network_alpha,
            max_train_epochs,
            save_every_n_epochs,
            seed,
            fp8_base,
            enable_lowvram,
            blocks_to_swap,
            attention,
            general_batch_size,
            general_resolutions,
            general_enable_bucket,
            general_bucket_no_upscale,
            image_resolutions,
            image_batch_size,
            image_enable_bucket,
            image_bucket_no_upscale,
            video_resolutions,
            video_batch_size,
            target_frames,
            frame_extraction,
            frame_stride,
            frame_sample,
            video_enable_bucket,
            video_bucket_no_upscale
        )
        
        if pid:
            # Disable the training button while training is active
            return message, pid, gr.update(visible=False), gr.update(visible=True)
        else:
            return message, pid, gr.update(visible=True), gr.update(visible=False)

    def handle_stop_click(pid):
        message = stop_training(pid)
        return message, gr.update(visible=True), gr.update(visible=False)

    def refresh_logs(log_box, pid):
        if pid is not None:
            return update_logs(log_box, pid)
        return log_box
    
    log_timer = gr.Timer(0.5, active=False) 
    
    log_timer.tick(
        fn=refresh_logs,
        inputs=[output, training_process_pid],
        outputs=output
    )
    
    def activate_timer():
        return gr.update(active=True)
    
    train_click = train_button.click(
        fn=handle_train_click,
        inputs=[
            dit_path, vae_path, llm_path, clip_path, dataset_path, config_dir, output_dir, dataset_name_input,
            mixed_precision_data_type, optimizer_type, lr, gradient_checkpointing, gradient_accumulation_steps, 
            max_data_loader_n_workers, persistent_data_loader_workers, network_dim, network_alpha,
            max_train_epochs, save_every, seed, fp8_base, enable_lowvram, blocks_to_swap,attention, general_batch_size, general_resolutions,
            general_enable_bucket, general_bucket_no_upscale, image_resolutions, image_batch_size,
            image_enable_bucket, image_bucket_no_upscale, video_resolutions, video_batch_size, target_frames,
            frame_extraction, frame_stride, frame_sample, video_enable_bucket, video_bucket_no_upscale
        ],
        outputs=[output, training_process_pid, train_button, stop_button],
        api_name=None
    ).then(
        fn=lambda: gr.update(active=True),  # Activate the Timer
        inputs=None,
        outputs=log_timer
    )
    
    def deactivate_timer():
        return gr.update(active=False)
    
    stop_click = stop_button.click(
        fn=handle_stop_click,
        inputs=[training_process_pid],
        outputs=[output,train_button, stop_button],
        api_name=None
    ).then(
        fn=lambda: gr.update(active=False),  # Deactivate the Timer
        inputs=None,
        outputs=log_timer
    )
    
    
    # Handle Download Button Click
    download_button.click(
        fn=handle_download,
        inputs=[dataset_path, download_options[0]],
        outputs=[download_zip, download_status],
        queue=True
    )
    
    with gr.Row():
        download_zip
    
    # Ensure that the "Download ZIP" button is only visible when a dataset is selected or created
    download_button.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=download_zip
    )
    
    # Handle selecting an existing dataset
    # existing_datasets.change(
    #     fn=handle_select_existing,
    #     inputs=existing_datasets,
    #     outputs=[
    #         dataset_path, 
    #         config_dir, 
    #         output_dir, 
    #         upload_status, 
    #         gallery,
    #         # download_button,
    #         download_status,
    #         hidden_config 
    #     ]
    # ).then(
    #     fn=lambda config_vals: update_ui_with_config(config_vals),
    #     inputs=hidden_config,  # Receives configuration values
    #     outputs=[
    #         epochs, batch_size, lr, save_every, eval_every, rank, only_double_blocks, dtype,
    #         transformer_path, vae_path, llm_path, clip_path, optimizer_type,
    #         betas, weight_decay, eps, gradient_accumulation_steps, num_repeats,
    #         resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, ar_buckets,
    #         frame_buckets, gradient_clipping, warmup_steps, eval_before_first_step,
    #         eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
    #         checkpoint_every_n_minutes, activation_checkpointing, partition_method,
    #         save_dtype, caching_batch_size, steps_per_print, video_clip_mode
    #     ]
    # )
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace", "."])
