# Chinese Audio Transcript Generator and Reviser

A Python script that processes audio files, generates transcripts using OpenAI's Whisper API, and then revises them using Claude 3.7 Sonnet to create well-structured articles.

## Features

- Splits audio files into 30-second chunks for efficient processing
- Transcribes audio using OpenAI's Whisper API (optimized for Chinese)
- Revises and reorganizes the raw transcript using Claude 3.7 Sonnet with "Thinking"
- Handles long transcripts by processing them in chunks of 1000 characters
- Maintains context between chunks for a coherent final article
- **Saves partial progress to a JSON file for debugging and recovery**
- **Allows resuming from a previous partial run**
- **Provides debugging options to print intermediate transcripts**
- **Supports parallel processing (default: 16 workers) for faster transcription and revision**
- **Supports domain-specific context documents for accurate jargon handling**
- **Uses parallel processing for faster audio chunking**
- **Implements automatic retry with exponential backoff for API rate limit handling**

## Requirements

- Python 3.8 or higher
- OpenAI API key (for Whisper)
- Anthropic API key (for Claude)
- FFmpeg (for audio processing)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install FFmpeg (required for pydub):
   - On macOS: `brew install ffmpeg`
   - On Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - On Windows: [Download FFmpeg](https://ffmpeg.org/download.html)

4. Set up your API keys by creating a `.env` file with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

Basic usage:

```
python transcript_generator.py path/to/your/audio_file.mp3
```

Recommend to add a context file containing the jargon, terminology and the background of the talk.

```
python transcript_generator.py path/to/your/audio_file.mp3 --context context.txt
```

Additional options:

```
python transcript_generator.py path/to/your/audio_file.mp3 \
  --output revised_transcript.txt \
  --language zh \
  --chunks_dir ./audio_chunks \
  --save_raw \
  --debug \
  --progress_file my_progress.json \
  --workers 16 \
  --context domain_jargon.txt \
  --max_retries 10
```

### Arguments

- `audio_path`: Path to the audio file (required)
- `--output`, `-o`: Output file path for the revised transcript (default: `revised_transcript.txt`)
- `--language`, `-l`: Language code for transcription (default: `zh` for Chinese)
- `--chunks_dir`: Directory to save the audio chunks (optional, uses a temp directory if not specified)
- `--save_raw`: Save the raw transcript before revision
- `--debug`: Print detailed debug information for each chunk during processing
- `--progress_file`: File to save and load partial progress (default: `transcript_progress.json`)
- `--no_progress`: Disable progress saving (don't create progress file)
- `--resume`: Resume processing from last saved progress (requires progress_file)
- `--workers`, `-w`: Number of parallel workers for transcription and revision (default: 16)
- `--context`, `-c`: Path to a context document (txt) containing domain-specific jargon/terminology
- `--max_retries`: Maximum number of retry attempts for API calls that encounter rate limits (default: 10)

## How It Works

1. **Parallel Audio Chunking**: The script divides the audio file into 30-second chunks using multiple threads, significantly speeding up the initial processing stage.

2. **Parallel Transcription**: Multiple audio chunks are transcribed simultaneously using OpenAI's Whisper API, greatly reducing processing time.

3. **Text Chunking**: The raw transcript is divided into chunks of approximately 1000 characters each, ensuring breaks occur at natural points in the text.

4. **Parallel Revision**: Multiple text chunks are revised simultaneously using Claude 3.7 Sonnet, significantly speeding up the process.

5. **Context-Aware Processing**: If provided with a domain-specific context document, Claude ensures proper handling of specialized jargon.

6. **Assembly**: The revised chunks are combined into a final, well-structured article.

7. **Progress Tracking**: At each step, the script can save progress to a JSON file, enabling recovery from interruptions or errors.

8. **Rate Limit Handling**: The script automatically handles API rate limits with exponential backoff and retries.

## Domain-Specific Context Documents

The script now supports using domain-specific context documents to improve the accuracy of specialized terminology in the revised transcript.

### How to Use Context Documents

1. Create a text file containing the domain-specific jargon, terminology and background of the talk (see `context.txt` in this repo for an example):
   ```
   直播主题：AI Agent，注定爆发？！

   时间：2025 年 3 月 13 日 20:00——22:00
   方式：极客公园微信视频号「今夜科技谈」直播（连麦）

   直播嘉宾：
   •	靖宇｜极客公园 副主编
   •	李博杰｜PINE AI 首席科学家
   •	宛辰｜极客公园 记者

   领域词汇表：
   Manus
   OpenAI Operator
   OpenAI Deep Research
   Anthropic
   Claude 3.7 Sonnet
   ...
   ```

2. Pass the context document to the script:
   ```
   python transcript_generator.py interview.mp3 --context context.txt
   ```

3. Claude will use this information to ensure that specialized terminology is correctly preserved and used in the revised transcript.

### Benefits of Context Documents

- Prevents misinterpretation of specialized terms
- Ensures consistent use of terminology throughout the document
- Improves the accuracy of technical content
- Useful for domains like medicine, law, finance, technology, etc.

## Rate Limit Handling

The script includes sophisticated rate limit handling to ensure reliable processing even when encountering API limits:

### Features

- **Automatic Retry**: When a rate limit error (HTTP 429) is encountered, the script automatically retries the API call.
- **Exponential Backoff**: Each retry uses an exponential backoff strategy, waiting progressively longer between attempts.
- **Random Jitter**: Adds random time variations to prevent all workers from retrying simultaneously.
- **Configurable Retry Count**: Set the maximum number of retry attempts with the `--max_retries` parameter.
- **Smart Worker Allocation**: Automatically reduces the number of parallel workers for API-intensive tasks to prevent rate limiting.
- **Progress Preservation**: All successfully processed chunks are saved immediately, ensuring no work is lost during retries.

### Example

```
python transcript_generator.py large_audio.mp3 --max_retries 15 --workers 8
```

This will:
1. Use a moderate number of parallel workers (8) to avoid hitting rate limits
2. Retry any rate-limited requests up to 15 times with increasing delays
3. Save all successfully processed chunks to enable resuming if needed

## Performance Optimization

The script uses parallel processing to significantly improve performance at every stage:

- **Parallel Audio Chunking**: Divides the audio file into chunks using multiple threads
- **ThreadPoolExecutor**: Uses Python's concurrent.futures for efficient parallel API calls and processing
- **Configurable Workers**: Adjust the number of parallel workers with the `--workers` flag
- **Thread-Safe Progress Tracking**: Uses locks to ensure thread-safe updates to the progress file
- **Efficient Resource Usage**: For most systems, 16 parallel workers provides a good balance between speed and resource usage
- **Rate-Limit Aware**: Automatically adjusts worker count for API-intensive tasks to prevent rate limiting

### Performance Comparison

|                  | Sequential Processing | Parallel Processing (16 workers) |
|------------------|----------------------|---------------------------------|
| Audio Chunking   | 100% (baseline)      | ~20-30% of baseline time         |
| Transcription    | 100% (baseline)      | ~6-10% of baseline time          |
| Revision         | 100% (baseline)      | ~6-10% of baseline time          |
| Total Processing | 100% (baseline)      | ~10-15% of baseline time         |

*Note: Actual performance gains depend on CPU cores, internet connection speed, and API response times.*

## Debugging and Recovery

The script provides several features to help with debugging and recovery:

- **Progress Saving**: Automatically saves progress after each chunk is processed (enabled by default)
- **Debug Mode**: With the `--debug` flag, the script will print each original chunk and its revised version
- **Resume Processing**: Use `--resume` to continue from the last saved point if the script was interrupted
- **Custom Progress File**: Specify the progress file location with `--progress_file`

### Example JSON Progress File

```json
{
  "transcriptions": ["chunk 1 text", "chunk 2 text", ...],
  "raw_transcript": "Full raw transcript text...",
  "chunks": ["text chunk 1", "text chunk 2", ...],
  "revised_chunks": ["revised chunk 1", "revised chunk 2", ...]
}
```

## Example

```
# Run with domain-specific terminology context
python transcript_generator.py medical_lecture.mp3 --context medical_terms.txt

# Run with maximum parallelism (32 workers) and domain context
python transcript_generator.py technical_talk.mp3 --workers 32 --context tech_jargon.txt

# If interrupted, resume from where it left off with context
python transcript_generator.py interview.mp3 --context industry_terms.txt --resume

# Process very large file with rate limit considerations
python transcript_generator.py long_conference.mp3 --workers 8 --max_retries 20
```

This will:
1. Split the audio file into 30-second chunks using parallel processing
2. Transcribe multiple chunks in parallel using Whisper
3. Save the raw transcript if requested
4. Process multiple transcript chunks in parallel with Claude, ensuring domain-specific terminology is handled correctly
5. Automatically retry any rate-limited API calls with exponential backoff
6. Save the final article to the specified output file
7. Save progress to the progress file after each step

## Limitations

- The quality of the final article depends on the quality of the original audio and the accuracy of the transcription.
- Processing long audio files can be time-consuming and may consume significant API credits.
- The script is optimized for Chinese, but can work with other languages by changing the `--language` parameter.
- Progress JSON files can become large with very long transcripts.
- Using too many parallel workers may lead to API rate limiting, especially with free or limited API plans.
- Parallel audio chunking requires sufficient RAM to hold the entire audio file in memory.

## License

[MIT License](LICENSE)

# Transcribe Utils

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Transcript Generation](#transcript-generation)
  - [PDF to PNG Conversion](#pdf-to-png-conversion)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## Overview

This repository contains utilities for transcription and document processing tasks.

## Features

### Transcript Generation
- Audio transcription using AI services
- Progress tracking with JSON state files
- Support for various audio formats

### PDF to PNG Conversion
- Export all pages of a PDF to PNG images
- Configurable resolution (default: 1920x1080)
- High-quality image output with precise scaling
- Maintains aspect ratio or exact dimensions
- Command-line interface with flexible options

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd transcribe-utils
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Transcript Generation

Use the `transcript_generator.py` script for audio transcription tasks.

### PDF to PNG Conversion

The `pdf_to_png.py` script converts PDF documents to high-quality PNG images.

#### Basic Usage

```bash
# Convert all pages to 1920x1080 PNG images
python pdf_to_png.py document.pdf

# Specify output directory
python pdf_to_png.py document.pdf output_folder

# Custom resolution
python pdf_to_png.py document.pdf --width 2560 --height 1440

# Show PDF information
python pdf_to_png.py document.pdf --info
```

#### Features

- **High-Quality Output**: Uses PyMuPDF for precise rendering control
- **Flexible Resolution**: Default 1920x1080 or custom dimensions
- **Smart Scaling**: Maintains aspect ratio or exact dimensions
- **Batch Processing**: Processes all pages automatically
- **Error Handling**: Continues processing if individual pages fail
- **Progress Tracking**: Shows conversion progress in real-time

#### Requirements

- **PyMuPDF**: For PDF processing and rendering
- **Pillow**: For advanced image processing and PNG optimization

#### Output

- PNG files named `page_0001.png`, `page_0002.png`, etc.
- Saved to `{pdf_name}_pages/` directory by default
- High-quality PNG with optimization

#### Examples

```bash
# Basic conversion
python pdf_to_png.py presentation.pdf
# Output: presentation_pages/page_0001.png, page_0002.png, ...

# Custom output directory and resolution  
python pdf_to_png.py document.pdf slides --width 3840 --height 2160

# Get PDF information first
python pdf_to_png.py document.pdf --info
``` 