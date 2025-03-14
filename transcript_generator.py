#!/usr/bin/env python3
"""
Audio Transcript Generator and Reviser

This script:
1. Splits an audio file into 30-second chunks
2. Transcribes each chunk using OpenAI's Whisper API
3. Combines the transcriptions
4. Uses Claude 3.7 Sonnet with thinking to revise the transcript into an organized article
5. Handles long transcripts by processing in chunks of 1000 characters
6. Saves partial progress to enable recovery from interruptions
7. Supports parallel processing for faster transcription and revision
8. Supports domain-specific context documents for accurate jargon handling
9. Uses parallel processing for faster audio chunking
10. Implements automatic retry for API rate limit errors
"""

import os
import time
import math
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import tempfile
import concurrent.futures
import threading
import random

# Audio processing
from pydub import AudioSegment

# API clients
import openai
from anthropic import Anthropic, APIError, APIStatusError, RateLimitError

# Load environment variables
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize API clients
openai.api_key = OPENAI_API_KEY
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Thread-safe lock for progress updates
progress_lock = threading.Lock()

# Maximum number of retry attempts for API calls
MAX_RETRY_ATTEMPTS = 10

def load_context_document(context_path: str) -> Optional[str]:
    """
    Load a context document containing domain-specific jargon/terminology
    
    Args:
        context_path: Path to the context document (txt format)
        
    Returns:
        Content of the context document or None if file doesn't exist/can't be read
    """
    if not context_path or not os.path.exists(context_path):
        print(f"Context document not found at: {context_path}")
        return None
        
    try:
        with open(context_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Loaded context document ({len(content)} characters) from: {context_path}")
        return content
    except Exception as e:
        print(f"Error loading context document: {e}")
        return None

def process_audio_chunk(args: Tuple[int, AudioSegment, int, int, str]) -> Tuple[int, str]:
    """
    Process a single audio chunk in parallel
    
    Args:
        args: Tuple containing (index, audio, start_ms, end_ms, output_dir)
        
    Returns:
        Tuple of (index, chunk_path)
    """
    index, audio, start_ms, end_ms, output_dir = args
    
    try:
        # Extract the chunk from the audio
        chunk = audio[start_ms:end_ms]
        
        # Create the chunk file path
        chunk_path = os.path.join(output_dir, f"chunk_{index:04d}.mp3")
        
        # Export the chunk
        chunk.export(chunk_path, format="mp3")
        
        return index, chunk_path
    except Exception as e:
        print(f"Error processing audio chunk {index}: {e}")
        return index, None

def chunk_audio(audio_path: str, chunk_size_ms: int = 30000, output_dir: str = None, max_workers: int = 16) -> List[str]:
    """
    Split audio file into chunks of specified size using parallel processing
    
    Args:
        audio_path: Path to the audio file
        chunk_size_ms: Size of each chunk in milliseconds (default 30 seconds)
        output_dir: Directory to save chunks (if None, uses temp directory)
        max_workers: Maximum number of parallel workers for chunking
        
    Returns:
        List of paths to the chunk files
    """
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total number of chunks
    total_chunks = math.ceil(len(audio) / chunk_size_ms)
    print(f"Splitting audio into {total_chunks} chunks of {chunk_size_ms/1000} seconds each (using {max_workers} parallel workers)")
    
    # Prepare tasks for parallel processing
    tasks = []
    for i in range(total_chunks):
        start_ms = i * chunk_size_ms
        end_ms = min((i + 1) * chunk_size_ms, len(audio))
        tasks.append((i, audio, start_ms, end_ms, output_dir))
    
    # Process chunks in parallel
    # We'll use ThreadPoolExecutor since we're sharing the audio object in memory
    chunk_paths = [None] * total_chunks  # Pre-allocate result list
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_audio_chunk, task) for task in tasks]
        
        # Process results as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Chunking audio"):
            try:
                index, chunk_path = future.result()
                if chunk_path:  # Only add if the chunk was processed successfully
                    chunk_paths[index] = chunk_path
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Filter out any None values (failed chunks)
    chunk_paths = [path for path in chunk_paths if path]
    
    return chunk_paths

def transcribe_audio_chunk(chunk_path: str, language: str = "zh") -> str:
    """
    Transcribe an audio chunk using OpenAI's Whisper API
    
    Args:
        chunk_path: Path to the audio chunk
        language: Language code (default "zh" for Chinese)
        
    Returns:
        Transcribed text
    """
    with open(chunk_path, "rb") as audio_file:
        try:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
            return response.text
        except Exception as e:
            print(f"Error transcribing {chunk_path}: {e}")
            time.sleep(1)  # Rate limit backoff
            return ""

def transcribe_chunk_with_index(args: Tuple[int, str, str, bool]) -> Tuple[int, str]:
    """
    Wrapper function for parallel transcription
    
    Args:
        args: Tuple containing (index, chunk_path, language, debug_print)
        
    Returns:
        Tuple of (index, transcribed_text)
    """
    index, chunk_path, language, debug_print = args
    
    print(f"Transcribing chunk {index+1}: {chunk_path}")
    transcription = transcribe_audio_chunk(chunk_path, language)
    
    # Print debug info if requested
    if debug_print:
        print(f"\n--- Transcription of chunk {index+1} ---")
        print(transcription)
        print("------------------------------\n")
    
    return index, transcription

def transcribe_audio_chunks(chunk_paths: List[str], language: str = "zh", debug_print: bool = False, 
                            progress_file: str = None, max_workers: int = 16) -> str:
    """
    Transcribe multiple audio chunks in parallel and combine the results
    
    Args:
        chunk_paths: List of paths to audio chunks
        language: Language code (default "zh" for Chinese)
        debug_print: Whether to print each chunk's transcription for debugging
        progress_file: File to save partial progress to
        max_workers: Maximum number of parallel workers for transcription
        
    Returns:
        Combined transcription text
    """
    print(f"Transcribing {len(chunk_paths)} audio chunks using Whisper API (with {max_workers} parallel workers)")
    
    # Track progress
    transcriptions = [""] * len(chunk_paths)  # Pre-allocate list with empty strings
    progress_data = {}
    
    # Load progress if available
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                if 'transcriptions' in progress_data and isinstance(progress_data['transcriptions'], list):
                    existing_transcriptions = progress_data['transcriptions']
                    # Copy existing transcriptions to our array
                    for i, trans in enumerate(existing_transcriptions):
                        if i < len(transcriptions):
                            transcriptions[i] = trans
                    print(f"Loaded {len(existing_transcriptions)} existing transcriptions from progress file")
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    # Find which chunks still need processing
    tasks = []
    for i, chunk_path in enumerate(chunk_paths):
        if i >= len(transcriptions) or not transcriptions[i]:
            tasks.append((i, chunk_path, language, debug_print))
    
    print(f"Processing {len(tasks)} remaining chunks")
    
    # Use ThreadPoolExecutor for network-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(transcribe_chunk_with_index, task) for task in tasks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                index, transcription = future.result()
                transcriptions[index] = transcription
                
                # Save progress atomically
                if progress_file:
                    with progress_lock:
                        with open(progress_file, 'r+' if os.path.exists(progress_file) else 'w', encoding='utf-8') as f:
                            try:
                                if os.path.exists(progress_file) and os.path.getsize(progress_file) > 0:
                                    progress_data = json.load(f)
                                else:
                                    progress_data = {}
                            except json.JSONDecodeError:
                                progress_data = {}
                            
                            progress_data['transcriptions'] = transcriptions
                            f.seek(0)
                            json.dump(progress_data, f, ensure_ascii=False, indent=2)
                            f.truncate()
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Filter out any empty strings
    transcriptions = [t for t in transcriptions if t]
    
    return " ".join(transcriptions)

def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks of approximately the specified size
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        
    Returns:
        List of text chunks
    """
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_position = 0
    
    while current_position < len(text):
        # Find a good breaking point near the chunk_size
        end_position = min(current_position + chunk_size, len(text))
        
        # Try to find a period, question mark, or exclamation mark to break at
        if end_position < len(text):
            for i in range(end_position, max(current_position, end_position - 200), -1):
                if text[i] in "。！？.!?":
                    end_position = i + 1
                    break
        
        chunks.append(text[current_position:end_position])
        current_position = end_position
    
    return chunks

def revise_transcript_with_claude(transcript_chunk: str, is_first_chunk: bool, is_last_chunk: bool, context_content: Optional[str] = None) -> str:
    """
    Revise a transcript chunk using Claude 3.7 Sonnet with thinking and automatic retry
    
    Args:
        transcript_chunk: The transcript chunk to revise
        is_first_chunk: Whether this is the first chunk in the series
        is_last_chunk: Whether this is the last chunk in the series
        context_content: Optional context document with domain-specific jargon
        
    Returns:
        Revised transcript text
    """
    # Prepare the system prompt based on chunk position and include context if provided
    base_system_prompt = "You are a professional editor revising a Chinese transcript."
    position_context = ""
    if is_first_chunk:
        position_context = "Organize this into the beginning of a well-structured article."
    elif is_last_chunk:
        position_context = "Organize this into the conclusion of a well-structured article."
    else:
        position_context = "Continue organizing this part of the transcript into a well-structured article."
    
    task_description = "Fix grammar, improve flow, and organize content logically."
    position_info = ""
    if is_first_chunk:
        position_info = "This is the first part of a longer transcript."
    elif is_last_chunk:
        position_info = "This is the last part of a longer transcript."
    else:
        position_info = "This is a middle section of a longer transcript."
    
    domain_context = ""
    if context_content:
        domain_context = f"""
Pay special attention to the following domain-specific jargon and terminology, ensuring they are properly used and maintained in the revised text:

{context_content}
"""

    system_prompt = f"{base_system_prompt} {position_context} {task_description} {position_info}{domain_context}"
    
    # Prepare the user prompt
    user_prompt = f"请根据以下原始中文转录内容，进行整理和修改，使其成为一篇组织良好的访谈稿片段。保持原意，但可以修正转录中的词句、语法错误，添加正确的标点符号，提高语言流畅度，并使内容更加连贯和有条理。请记住这是一篇访谈稿的片段，需要保持对话的连贯性，保持内容原意，不要变成内容总结，不要添加标题。访谈稿的中文和英文单词之间需要有空格。被访者、主持人的名字在背景介绍材料中有，他们的名字需要用粗体显示。"
    
    if context_content:
        user_prompt += f"\n\n请确保正确使用与保留领域内的专业术语和行话。"
    
    user_prompt += f"\n\n原始转录内容中的一个片段：\n{transcript_chunk}"
    
    # Implement retry logic with exponential backoff
    retry_count = 0
    while retry_count < MAX_RETRY_ATTEMPTS:
        try:
            # Call Claude API with thinking enabled
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            # Extract the response content
            return response.content[0].text
            
        except RateLimitError as e:
            retry_count += 1
            # Calculate exponential backoff with jitter
            # Start with 10 seconds, then 20, 40, 80, etc. plus random jitter
            backoff_seconds = (2 ** retry_count) * 5 + random.uniform(0, 5)
            
            # If this is the last retry, raise the exception
            if retry_count >= MAX_RETRY_ATTEMPTS:
                print(f"Failed after {MAX_RETRY_ATTEMPTS} attempts. Last error: {e}")
                return transcript_chunk  # Return original if all retries failed
            
            print(f"Rate limit exceeded (attempt {retry_count}/{MAX_RETRY_ATTEMPTS}). "
                  f"Retrying in {backoff_seconds:.2f} seconds...")
            time.sleep(backoff_seconds)
            
        except APIStatusError as e:
            if e.status_code == 429:  # Another way rate limits might be reported
                retry_count += 1
                backoff_seconds = (2 ** retry_count) * 5 + random.uniform(0, 5)
                
                if retry_count >= MAX_RETRY_ATTEMPTS:
                    print(f"Failed after {MAX_RETRY_ATTEMPTS} attempts. Last error: {e}")
                    return transcript_chunk
                
                print(f"Rate limit exceeded (attempt {retry_count}/{MAX_RETRY_ATTEMPTS}). "
                      f"Retrying in {backoff_seconds:.2f} seconds...")
                time.sleep(backoff_seconds)
            else:
                print(f"API error: {e}")
                time.sleep(2)  # Brief pause before returning
                return transcript_chunk  # Return original for non-rate-limit errors
                
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            time.sleep(2)  # Brief pause before returning
            return transcript_chunk  # Return original if API call fails
    
    # This should never be reached due to the return in the last retry, but just in case
    return transcript_chunk

def revise_chunk_with_metadata(args: Tuple[int, str, bool, bool, bool, Optional[str]]) -> Tuple[int, str]:
    """
    Wrapper function for parallel revision
    
    Args:
        args: Tuple containing (index, chunk, is_first, is_last, debug_print, context_content)
        
    Returns:
        Tuple of (index, revised_text)
    """
    index, chunk, is_first, is_last, debug_print, context_content = args
    
    print(f"Revising chunk {index+1}")
    
    # Print original chunk if debug is enabled
    if debug_print:
        print(f"\n--- Original chunk {index+1} ---")
        print(chunk)
        print("----------------------------\n")
    
    # Revise the chunk
    revised_chunk = revise_transcript_with_claude(chunk, is_first, is_last, context_content)
    
    # Print revised chunk if debug is enabled
    if debug_print:
        print(f"\n--- Revised chunk {index+1} ---")
        print(revised_chunk)
        print("----------------------------\n")
    
    return index, revised_chunk

def process_transcript(transcript: str, debug_print: bool = False, progress_file: str = None, 
                       max_workers: int = 16, context_content: Optional[str] = None) -> str:
    """
    Process a full transcript by splitting it into chunks and revising each chunk in parallel
    
    Args:
        transcript: The full transcript text
        debug_print: Whether to print debug information
        progress_file: File to save partial progress to
        max_workers: Maximum number of parallel workers for revision
        context_content: Optional context document with domain-specific jargon
        
    Returns:
        Revised transcript as an organized article
    """
    # Split the transcript into manageable chunks
    print("Splitting transcript into chunks for processing with Claude")
    chunks = split_text_into_chunks(transcript)
    print(f"Split transcript into {len(chunks)} chunks (will use {max_workers} parallel workers)")
    
    # Initialize progress data
    progress_data = {}
    revised_chunks = [""] * len(chunks)  # Pre-allocate with empty strings
    
    # Load progress if available
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                if 'revised_chunks' in progress_data and isinstance(progress_data['revised_chunks'], list):
                    existing_revisions = progress_data['revised_chunks']
                    # Copy existing revisions to our array
                    for i, rev in enumerate(existing_revisions):
                        if i < len(revised_chunks):
                            revised_chunks[i] = rev
                    print(f"Loaded {len(existing_revisions)} existing revised chunks from progress file")
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    # Find which chunks still need processing
    tasks = []
    for i, chunk in enumerate(chunks):
        if i >= len(revised_chunks) or not revised_chunks[i]:
            is_first = (i == 0)
            is_last = (i == len(chunks) - 1)
            tasks.append((i, chunk, is_first, is_last, debug_print, context_content))
    
    print(f"Processing {len(tasks)} remaining chunks")
    
    # Set a reduced number of workers specifically for API calls to avoid rate limits
    # Default to half the requested workers, with a minimum of 2 and maximum of 8
    api_workers = min(max(max_workers // 2, 2), 8)
    print(f"Using {api_workers} workers for Claude API calls to avoid rate limits")
    
    # Use ThreadPoolExecutor for network-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=api_workers) as executor:
        futures = [executor.submit(revise_chunk_with_metadata, task) for task in tasks]
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Revising chunks"):
            try:
                index, revised_chunk = future.result()
                revised_chunks[index] = revised_chunk
                
                # Save progress atomically
                if progress_file:
                    with progress_lock:
                        with open(progress_file, 'r+' if os.path.exists(progress_file) else 'w', encoding='utf-8') as f:
                            try:
                                if os.path.exists(progress_file) and os.path.getsize(progress_file) > 0:
                                    progress_data = json.load(f)
                                else:
                                    progress_data = {}
                            except json.JSONDecodeError:
                                progress_data = {}
                            
                            progress_data['chunks'] = chunks
                            progress_data['revised_chunks'] = revised_chunks
                            f.seek(0)
                            json.dump(progress_data, f, ensure_ascii=False, indent=2)
                            f.truncate()
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Filter out any empty strings
    revised_chunks = [c for c in revised_chunks if c]
    
    # Combine the revised chunks
    return "\n\n".join(revised_chunks)

def main():
    parser = argparse.ArgumentParser(description="Audio Transcript Generator and Reviser")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--output", "-o", default="revised_transcript.txt", help="Output file path")
    parser.add_argument("--language", "-l", default="zh", help="Language code (default: zh for Chinese)")
    parser.add_argument("--chunks_dir", help="Directory to save audio chunks (optional)")
    parser.add_argument("--save_raw", action="store_true", help="Save the raw transcript before revision")
    parser.add_argument("--debug", action="store_true", help="Print debug information during processing")
    parser.add_argument("--progress_file", default="transcript_progress.json", 
                       help="File to save and load partial progress (default: transcript_progress.json)")
    parser.add_argument("--no_progress", action="store_true", 
                       help="Disable progress saving (don't create progress file)")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from last saved progress (requires progress_file)")
    parser.add_argument("--workers", "-w", type=int, default=16,
                       help="Number of parallel workers (default: 16)")
    parser.add_argument("--context", "-c", 
                       help="Path to a context document (txt) containing domain-specific jargon/terminology")
    parser.add_argument("--max_retries", type=int, default=10,
                       help="Maximum number of retry attempts for API calls (default: 10)")
    
    args = parser.parse_args()
    
    # Set global retry count
    global MAX_RETRY_ATTEMPTS
    MAX_RETRY_ATTEMPTS = args.max_retries
    
    # Check if API keys are set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set")
        return
    
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        return
    
    # Determine progress file path
    progress_file = None if args.no_progress else args.progress_file
    
    # Load context document if provided
    context_content = None
    if args.context:
        context_content = load_context_document(args.context)
    
    # Process the audio file
    start_time = time.time()
    
    # Load existing progress if resuming
    raw_transcript = None
    if args.resume and progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                if 'raw_transcript' in progress_data:
                    raw_transcript = progress_data['raw_transcript']
                    print(f"Loaded raw transcript from progress file ({len(raw_transcript)} characters)")
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    # If we don't have a raw transcript yet, process the audio
    if not raw_transcript:
        # Step 1: Chunk the audio in parallel
        chunk_paths = chunk_audio(
            args.audio_path, 
            output_dir=args.chunks_dir,
            max_workers=args.workers
        )
        
        # Step 2: Transcribe the chunks in parallel
        raw_transcript = transcribe_audio_chunks(
            chunk_paths, 
            args.language, 
            debug_print=args.debug, 
            progress_file=progress_file,
            max_workers=args.workers
        )
        
        # Save progress after transcription
        if progress_file:
            with open(progress_file, 'r+' if os.path.exists(progress_file) else 'w', encoding='utf-8') as f:
                progress_data = json.load(f) if os.path.exists(progress_file) and os.path.getsize(progress_file) > 0 else {}
                progress_data['raw_transcript'] = raw_transcript
                f.seek(0)
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
                f.truncate()
    
    # Save raw transcript if requested
    if args.save_raw:
        raw_output_path = f"raw_{args.output}"
        with open(raw_output_path, "w", encoding="utf-8") as f:
            f.write(raw_transcript)
        print(f"Raw transcript saved to {raw_output_path}")
    
    # Step 3: Revise the transcript with Claude in parallel
    revised_transcript = process_transcript(
        raw_transcript, 
        debug_print=args.debug, 
        progress_file=progress_file,
        max_workers=args.workers,
        context_content=context_content
    )
    
    # Save the revised transcript
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(revised_transcript)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Revised transcript saved to {args.output}")

if __name__ == "__main__":
    main()