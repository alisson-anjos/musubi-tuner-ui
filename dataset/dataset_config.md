## Dataset Configuration

Please create a TOML file for dataset configuration.

Image and video datasets are supported. The configuration file can include multiple datasets, either image or video datasets, with caption text files or metadata JSONL files.

The cache directory must be different for each dataset.

### Sample for Image Dataset with Caption Text Files

```toml
# resolution, caption_extension, batch_size, enable_bucket, bucket_no_upscale must be set in either general or datasets

# general configurations
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/path/to/image_dir"
cache_directory = "/path/to/cache_directory"

# other datasets can be added here. each dataset can have different configurations
```

`cache_directory` is optional, default is None to use the same directory as the image directory. However, we recommend to set the cache directory to avoid accidental sharing of the cache files between different datasets.

### Sample for Image Dataset with Metadata JSONL File

```toml
# resolution, batch_size, enable_bucket, bucket_no_upscale must be set in either general or datasets
# caption_extension is not required for metadata jsonl file
# cache_directory is required for each dataset with metadata jsonl file

# general configurations
[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl"
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file

# other datasets can be added here. each dataset can have different configurations
```

JSONL file format for metadata:

```json
{"image_path": "/path/to/image1.jpg", "caption": "A caption for image1"}
{"image_path": "/path/to/image2.jpg", "caption": "A caption for image2"}
```

### Sample for Video Dataset with Caption Text Files

```toml
# resolution, caption_extension, target_frames, frame_extraction, frame_stride, frame_sample, batch_size, enable_bucket, bucket_no_upscale must be set in either general or datasets

# general configurations
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "/path/to/video_dir"
cache_directory = "/path/to/cache_directory" # recommended to set cache directory
target_frames = [1, 25, 45]
frame_extraction = "head"

# other datasets can be added here. each dataset can have different configurations
```

### Sample for Video Dataset with Metadata JSONL File

```toml
# resolution, target_frames, frame_extraction, frame_stride, frame_sample, batch_size, enable_bucket, bucket_no_upscale must be set in either general or datasets
# caption_extension is not required for metadata jsonl file
# cache_directory is required for each dataset with metadata jsonl file

# general configurations
[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl"
target_frames = [1, 25, 45]
frame_extraction = "head"
cache_directory = "/path/to/cache_directory_head"

# same metadata jsonl file can be used for multiple datasets
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl"
target_frames = [1]
frame_stride = 10
cache_directory = "/path/to/cache_directory_stride"

# other datasets can be added here. each dataset can have different configurations
```

JSONL file format for metadata:

```json
{"video_path": "/path/to/video1.mp4", "caption": "A caption for video1"}
{"video_path": "/path/to/video2.mp4", "caption": "A caption for video2"}
```

### frame_extraction Options

- `head`: Extract the first N frames from the video.
- `chunk`: Extract frames by splitting the video into chunks of N frames.
- `slide`: Extract frames from the video with a stride of `frame_stride`.
- `uniform`: Extract `frame_sample` samples uniformly from the video.

For example, consider a video with 40 frames. The following diagrams illustrate each extraction:

```
Original Video, 40 frames: x = frame, o = no frame
oooooooooooooooooooooooooooooooooooooooo

head, target_frames = [1, 13, 25] -> extract head frames:
xooooooooooooooooooooooooooooooooooooooo
xxxxxxxxxxxxxooooooooooooooooooooooooooo
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo

chunk, target_frames = [13, 25] -> extract frames by splitting into chunks, into 13 and 25 frames:
xxxxxxxxxxxxxooooooooooooooooooooooooooo
oooooooooooooxxxxxxxxxxxxxoooooooooooooo
ooooooooooooooooooooooooooxxxxxxxxxxxxxo
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo

NOTE: Please do not include 1 in target_frames if you are using the frame_extraction "chunk". It will make the all frames to be extracted.

slide, target_frames = [1, 13, 25], frame_stride = 10 -> extract N frames with a stride of 10:
xooooooooooooooooooooooooooooooooooooooo
ooooooooooxooooooooooooooooooooooooooooo
ooooooooooooooooooooxooooooooooooooooooo
ooooooooooooooooooooooooooooooxooooooooo
xxxxxxxxxxxxxooooooooooooooooooooooooooo
ooooooooooxxxxxxxxxxxxxooooooooooooooooo
ooooooooooooooooooooxxxxxxxxxxxxxooooooo
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo
ooooooooooxxxxxxxxxxxxxxxxxxxxxxxxxooooo

uniform, target_frames =[1, 13, 25], frame_sample = 4 -> extract `frame_sample` samples uniformly, N frames each:
xooooooooooooooooooooooooooooooooooooooo
oooooooooooooxoooooooooooooooooooooooooo
oooooooooooooooooooooooooxoooooooooooooo
ooooooooooooooooooooooooooooooooooooooox
xxxxxxxxxxxxxooooooooooooooooooooooooooo
oooooooooxxxxxxxxxxxxxoooooooooooooooooo
ooooooooooooooooooxxxxxxxxxxxxxooooooooo
oooooooooooooooooooooooooooxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo
oooooxxxxxxxxxxxxxxxxxxxxxxxxxoooooooooo
ooooooooooxxxxxxxxxxxxxxxxxxxxxxxxxooooo
oooooooooooooooxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Specifications

```toml
# general configurations
[general]
resolution = [960, 544] # optional, [W, H], default is None. This is the default resolution for all datasets
caption_extension = ".txt" # optional, default is None. This is the default caption extension for all datasets
batch_size = 1 # optional, default is 1. This is the default batch size for all datasets
enable_bucket = true # optional, default is false. Enable bucketing for datasets
bucket_no_upscale = false # optional, default is false. Disable upscaling for bucketing. Ignored if enable_bucket is false

### Image Dataset

# sample image dataset with caption text files
[[datasets]]
image_directory = "/path/to/image_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
batch_size = 4 # optional, overwrite the default batch size
enable_bucket = false # optional, overwrite the default bucketing setting
bucket_no_upscale = true # optional, overwrite the default bucketing setting
cache_directory = "/path/to/cache_directory" # optional, default is None to use the same directory as the image directory. NOTE: caching is always enabled

# sample image dataset with metadata **jsonl** file
[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of image files and captions
resolution = [960, 544] # required if general resolution is not set
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# caption_extension is not required for metadata jsonl file
# batch_size, enable_bucket, bucket_no_upscale are also available for metadata jsonl file

### Video Dataset

# sample video dataset with caption text files
[[datasets]]
video_directory = "/path/to/video_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set

target_frames = [1, 25, 79] # required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)

# NOTE: Please do not include 1 in target_frames if you are using the frame_extraction "chunk". It will make the all frames to be extracted.

frame_extraction = "head" # optional, "head" or "chunk", "slide", "uniform". Default is "head"
frame_stride = 1 # optional, default is 1, available for "slide" frame extraction
frame_sample = 4 # optional, default is 1 (same as "head"), available for "uniform" frame extraction
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for video dataset

# sample video dataset with metadata jsonl file
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of video files and captions

target_frames = [1, 79]

cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# frame_extraction, frame_stride, frame_sample are also available for metadata jsonl file
```

<!-- 
# sample image dataset with lance
[[datasets]]
image_lance_dataset = "/path/to/lance_dataset"
resolution = [960, 544] # required if general resolution is not set
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for lance dataset
-->

The metadata with .json file will be supported in the near future.



<!--

```toml
# general configurations
[general]
resolution = [960, 544] # optional, [W, H], default is None. This is the default resolution for all datasets
caption_extension = ".txt" # optional, default is None. This is the default caption extension for all datasets
batch_size = 1 # optional, default is 1. This is the default batch size for all datasets
enable_bucket = true # optional, default is false. Enable bucketing for datasets
bucket_no_upscale = false # optional, default is false. Disable upscaling for bucketing. Ignored if enable_bucket is false

# sample image dataset with caption text files
[[datasets]]
image_directory = "/path/to/image_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
batch_size = 4 # optional, overwrite the default batch size
enable_bucket = false # optional, overwrite the default bucketing setting
bucket_no_upscale = true # optional, overwrite the default bucketing setting
cache_directory = "/path/to/cache_directory" # optional, default is None to use the same directory as the image directory. NOTE: caching is always enabled

# sample image dataset with metadata **jsonl** file
[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of image files and captions
resolution = [960, 544] # required if general resolution is not set
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# caption_extension is not required for metadata jsonl file
# batch_size, enable_bucket, bucket_no_upscale are also available for metadata jsonl file

# sample video dataset with caption text files
[[datasets]]
video_directory = "/path/to/video_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
target_frames = [1, 25, 79] # required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)
frame_extraction = "head" # optional, "head" or "chunk", "slide", "uniform". Default is "head"
frame_stride = 1 # optional, default is 1, available for "slide" frame extraction
frame_sample = 4 # optional, default is 1 (same as "head"), available for "uniform" frame extraction
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for video dataset

# sample video dataset with metadata jsonl file
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of video files and captions
target_frames = [1, 79]
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# frame_extraction, frame_stride, frame_sample are also available for metadata jsonl file
```

# sample image dataset with lance
[[datasets]]
image_lance_dataset = "/path/to/lance_dataset"
resolution = [960, 544] # required if general resolution is not set
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for lance dataset

The metadata with .json file will be supported in the near future.




-->