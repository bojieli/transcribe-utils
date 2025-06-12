#!/usr/bin/env python3
"""
PDF to PNG Converter

This script converts all pages of a PDF file to PNG images with a specified resolution (1920x1080).
It uses PyMuPDF (fitz) for precise control over image resolution and quality.

Usage:
    python pdf_to_png.py input.pdf [output_directory]

Requirements:
    - PyMuPDF (fitz)
    - Pillow (PIL)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import pymupdf as fitz
except ImportError:
    print("Error: PyMuPDF is not installed. Please install it with: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is not installed. Please install it with: pip install Pillow")
    sys.exit(1)


class PDFToPNGConverter:
    """Converts PDF pages to PNG images with specified resolution."""
    
    def __init__(self, target_width: int = 1920, target_height: int = 1080):
        """
        Initialize converter with target resolution.
        
        Args:
            target_width: Target width in pixels
            target_height: Target height in pixels
        """
        self.target_width = target_width
        self.target_height = target_height
    
    def convert_pdf_to_png(self, pdf_path: str, output_dir: Optional[str] = None) -> None:
        """
        Convert all pages of a PDF to PNG files.
        
        Args:
            pdf_path: Path to the input PDF file
            output_dir: Directory to save PNG files (optional)
        """
        # Validate input file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = Path(pdf_path).stem + "_pages"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Open PDF document
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise Exception(f"Failed to open PDF: {e}")
        
        print(f"Converting {len(doc)} pages from '{pdf_path}'...")
        print(f"Target resolution: {self.target_width}x{self.target_height}")
        print(f"Output directory: {output_dir}")
        
        # Process each page
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # Get page dimensions
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                
                # Calculate scaling factors to achieve target resolution
                scale_x = self.target_width / page_width
                scale_y = self.target_height / page_height
                
                # Use the smaller scale factor to maintain aspect ratio
                # or use different scales for exact target dimensions
                scale = min(scale_x, scale_y)  # Maintain aspect ratio
                
                # Create transformation matrix
                mat = fitz.Matrix(scale, scale)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL Image for better PNG handling
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # If we want exact dimensions, resize the image
                if img.size != (self.target_width, self.target_height):
                    img = img.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
                
                # Save as PNG
                output_filename = f"page_{page_num + 1:04d}.png"
                output_path = os.path.join(output_dir, output_filename)
                img.save(output_path, "PNG", optimize=True)
                
                print(f"  ✓ Page {page_num + 1}/{len(doc)} -> {output_filename}")
                
                # Clean up
                pix = None
                img.close()
                
            except Exception as e:
                print(f"  ✗ Error processing page {page_num + 1}: {e}")
                continue
        
        # Close document
        doc.close()
        print(f"\nConversion complete! Files saved to: {output_dir}")
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get information about the PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        info = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", "Unknown"),
            "author": doc.metadata.get("author", "Unknown"),
            "pages": []
        }
        
        # Get information for each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            info["pages"].append({
                "page_number": page_num + 1,
                "width": page_rect.width,
                "height": page_rect.height,
                "rotation": page.rotation
            })
        
        doc.close()
        return info


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to PNG images with 1920x1080 resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_png.py document.pdf
  python pdf_to_png.py document.pdf output_folder
  python pdf_to_png.py document.pdf --width 1920 --height 1080
  python pdf_to_png.py document.pdf --info
        """
    )
    
    parser.add_argument("pdf_file", help="Path to the input PDF file")
    parser.add_argument("output_dir", nargs="?", help="Output directory for PNG files")
    parser.add_argument("--width", type=int, default=1920, help="Target width in pixels (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Target height in pixels (default: 1080)")
    parser.add_argument("--info", action="store_true", help="Show PDF information only")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file '{args.pdf_file}' not found.")
        sys.exit(1)
    
    if not args.pdf_file.lower().endswith('.pdf'):
        print(f"Warning: File '{args.pdf_file}' doesn't appear to be a PDF file.")
    
    # Create converter
    converter = PDFToPNGConverter(args.width, args.height)
    
    try:
        if args.info:
            # Show PDF information
            info = converter.get_pdf_info(args.pdf_file)
            print(f"\nPDF Information:")
            print(f"  File: {args.pdf_file}")
            print(f"  Title: {info['title']}")
            print(f"  Author: {info['author']}")
            print(f"  Pages: {info['page_count']}")
            print(f"\nPage Details:")
            for page_info in info['pages']:
                print(f"  Page {page_info['page_number']}: "
                      f"{page_info['width']:.1f}x{page_info['height']:.1f} pts, "
                      f"rotation: {page_info['rotation']}°")
        else:
            # Convert PDF to PNG
            converter.convert_pdf_to_png(args.pdf_file, args.output_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import io  # Import needed for PIL Image handling
    main() 