# Standard imports
import io
import os
import shutil
import subprocess
import tempfile
import time
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import csv
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger('sigmafyai')


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def upload_image_to_imgbb(image_data, filename):
    """Upload image data to imgbb temporary hosting service
    
    Args:
        image_data: Image data as bytes
        filename: Name for the uploaded file
        
    Returns:
        tuple: (hosted_url, error message or None)
    """
    try:
        import requests
        
        # imgbb API endpoint (free service with temporary hosting)
        api_url = "https://api.imgbb.com/1/upload"
        
        # You can get a free API key from https://api.imgbb.com/
        # For demo purposes, we'll use a public key (replace with your own)
        api_key = os.getenv("IMGBB_API_KEY") # Replace with actual API key
        
        if not api_key:
            return None, "IMGBB_API_KEY not set in environment variables"
        
        # Convert image data to base64 for upload
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare the payload
        payload = {
            'key': api_key,
            'image': image_b64,
            'name': filename,
            'expiration': 600  # 10 minutes expiration
        }
        
        # Upload the image
        response = requests.post(api_url, data=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                hosted_url = result['data']['url']
                logger.info(f"✓ Image successfully uploaded to ImgBB: {hosted_url}")
                return hosted_url, None
            else:
                return None, f"Upload failed: {result.get('error', {}).get('message', 'Unknown error')}"
        else:
            return None, f"HTTP error: {response.status_code}"
            
    except ImportError:
        return None, "requests library not installed. Run: pip install requests"
    except Exception as e:
        return None, f"Image upload error: {str(e)}"


def upload_image_to_temp_storage(image_data, filename):
    """Upload image to temporary storage with automatic cleanup
    
    This function tries multiple approaches:
    1. Use imgbb for temporary hosting (requires API key)
    2. Fall back to base64 encoding for direct API use
    
    Args:
        image_data: Image data as bytes
        filename: Name for the uploaded file
        
    Returns:
        tuple: (image_reference, error message or None)
        image_reference can be either a URL or base64 data URL
    """
    # Try imgbb first (if API key is configured)
    hosted_url, error = upload_image_to_imgbb(image_data, filename)
    
    if hosted_url:
        return hosted_url, None
    
    # Log why imgbb failed
    logger.warning(f"⚠ ImgBB upload failed: {error}")
    logger.info("📝 Falling back to base64 data URL...")
    
    # Fall back to base64 data URL (works directly with OpenAI API)
    try:
        import base64
        
        # Validate image data size (OpenAI has limits)
        max_size = 20 * 1024 * 1024  # 20MB limit
        if len(image_data) > max_size:
            return None, f"Image too large: {len(image_data)} bytes (max: {max_size})"
        
        # Determine image format from filename - ONLY use formats supported by OpenAI Vision API
        file_ext = filename.lower().split('.')[-1]
        if file_ext in ['jpg', 'jpeg']:
            mime_type = 'image/jpeg'
        elif file_ext == 'png':
            mime_type = 'image/png'
        elif file_ext == 'webp':
            mime_type = 'image/webp'
        elif file_ext == 'gif':
            mime_type = 'image/gif'
        else:
            # Always default to PNG for OpenAI Vision API compatibility
            mime_type = 'image/png'
            logger.warning(f"⚠ Converting image to PNG format for OpenAI Vision API compatibility")
        
        # Create base64 data URL
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:{mime_type};base64,{image_b64}"
        
        # Validate the data URL format
        if not data_url.startswith('data:image/'):
            return None, "Invalid data URL format generated"
        
        logger.info(f"✓ Created base64 data URL ({len(data_url)} characters, {mime_type})")
        return data_url, None
        
    except Exception as e:
        return None, f"Image processing error: {str(e)}"


def convert_pdf_to_images(pdf_content, output_dir="extracted_output"):
    """Convert PDF content to PNG images and upload to temporary storage
    
    Args:
        pdf_content: PDF file content as bytes
        output_dir: Directory to save the extracted images (for backup)
        
    Returns:
        tuple: (list of image references, error message or None)
        image references can be URLs or base64 data URLs
    """
    try:
        # Convert PDF to images using pdf2image with lower DPI for smaller file size
        images = convert_from_bytes(pdf_content, dpi=72, fmt='PNG')  # Reduced DPI from 150 to 72
        
        if not images:
            return None, "No images could be extracted from PDF"
        
        image_references = []
        
        for i, img in enumerate(images):
            try:
                # Ensure image is in RGB mode (required for PNG)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert image to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG', optimize=True)
                img_data = img_buffer.getvalue()
                
                # Upload image to temporary storage
                filename = f"page_{i+1}.png"
                image_ref, upload_error = upload_image_to_temp_storage(img_data, filename)
                
                if upload_error:
                    print(f"Error uploading page {i+1}: {upload_error}")
                    continue
                
                image_references.append(image_ref)
                # Only print URL if it's not a base64 data URL (to avoid cluttering output)
                if image_ref.startswith('http'):
                    print(f"Hosted page {i+1} at {image_ref}")
                else:
                    print(f"Processed page {i+1} as base64 data URL")
                
            except Exception as page_error:
                print(f"Error processing page {i+1}: {page_error}")
                continue
        
        if not image_references:
            return None, "No images could be successfully processed"
            
        return image_references, None
        
    except Exception as e:
        return None, f"PDF conversion error: {str(e)}"


def preprocess_google_drive_url(url):
    """
    Convert Google Drive/Sheets sharing URL to direct download URL
    """
    if "docs.google.com/spreadsheets" in url:
        # Handle Google Sheets URLs
        if "/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    elif "drive.google.com" in url:
        # Handle Google Drive file URLs
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def preprocess_google_drive_url_v2(url):
    """
    Convert Google Drive/Sheets sharing URL to direct download URL - bekh
    """
    # TODO: remove unnecessary keyword
    try:
        logger.info(f"Processing URL: {url}")
        
        if "docs.google.com/spreadsheets" in url:
            # Handle Google Sheets URLs
            if "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
                processed_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                logger.info(f"Converted Google Sheets URL to: {processed_url}")
                return processed_url
        elif "drive.google.com" in url:
            # Handle Google Drive file URLs
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
                # Use direct file URL that works better with external APIs
                processed_url = f"https://drive.google.com/file/d/{file_id}/preview"
                logger.info(f"Converted Google Drive URL to preview URL: {processed_url}")
                return processed_url
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
                processed_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                logger.info(f"Converted Google Drive URL to: {processed_url}")
                return processed_url
        
        # If no conversion was done, return the original URL
        return url
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return url


def load_csv_robust(csv_input):
    """
    Load CSV from URL or file with robust error handling and preprocessing
    """
    # Handle file uploads vs URLs
    if hasattr(csv_input, 'read'):  # File-like object
        try:
            df = pd.read_csv(csv_input, on_bad_lines='skip')
        except Exception:
            # Try with more lenient settings
            csv_input.seek(0)  # Reset file pointer
            df = pd.read_csv(csv_input, on_bad_lines='skip', quoting=1, skipinitialspace=True)
    else:  # URL string
        # Preprocess Google Drive URLs
        processed_url = preprocess_google_drive_url(csv_input)

        # Try multiple parsing strategies
        strategies = [
            # Strategy 1: Standard parsing
            {'on_bad_lines': 'skip'},
            # Strategy 2: More lenient parsing
            {'on_bad_lines': 'skip', 'quoting': 1, 'skipinitialspace': True},
            # Strategy 3: Very lenient parsing
            {'on_bad_lines': 'skip', 'sep': ',', 'quoting': 3, 'skipinitialspace': True, 'encoding': 'utf-8'},
            # Strategy 4: Try with different separator
            {'on_bad_lines': 'skip', 'sep': ';', 'quoting': 1},
        ]

        df = None
        for i, strategy in enumerate(strategies, 1):
            try:
                df = pd.read_csv(processed_url, **strategy)
                break
            except Exception as e:
                # Continue to next strategy
                continue

        # If all strategies fail, try manual preprocessing
        if df is None:
            try:
                response = requests.get(processed_url)
                content = response.text

                # Clean up the content
                lines = content.split('\n')

                # Find the header and determine expected column count
                header_line = None
                expected_cols = 0
                for i, line in enumerate(lines[:10]):  # Check first 10 lines
                    if line.strip() and ',' in line:
                        cols = len(line.split(','))
                        if cols > expected_cols:
                            header_line = i
                            expected_cols = cols

                if header_line is not None:
                    # Filter lines with consistent column count
                    clean_lines = []
                    for i, line in enumerate(lines):
                        if not line.strip():  # Skip empty lines
                            continue
                        cols = len(line.split(','))
                        if cols == expected_cols or i <= header_line:  # Keep header and consistent lines
                            clean_lines.append(line)

                    # Try to parse cleaned content
                    clean_content = '\n'.join(clean_lines)
                    df = pd.read_csv(io.StringIO(clean_content), on_bad_lines='skip')
            except Exception:
                pass

        if df is None:
            raise Exception("Failed to parse CSV with all available strategies")

    # Clean the DataFrame
    df = df.dropna(how='all')  # Remove rows where all values are NaN
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    df = df.loc[:, df.notna().any()]  # Remove columns that are all NaN

    # Remove columns that are completely empty or have no meaningful data
    columns_to_remove = []
    for col in df.columns:
        # Check if column has any non-null values
        non_null_count = df[col].notna().sum()
        total_count = len(df)

        # Only remove if completely empty or has less than 2 non-null values
        if non_null_count == 0:
            columns_to_remove.append(col)
        elif non_null_count < 2 and total_count > 10:  # For larger datasets, need at least 2 values
            columns_to_remove.append(col)

    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)

    # Remove trailing empty columns that might have been missed
    while len(df.columns) > 0 and df.iloc[:, -1].isnull().all():
        df = df.drop(columns=[df.columns[-1]])

    return df


def extract_text_from_pdf(pdf_content):
    """Extract text content from PDF using PyPDF2"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        return text_content.strip(), None
    except Exception as e:
        return None, str(e)


def generate_graph_with_llm(data, graph_type):
    csv_data = data.to_csv(index=False)

    system_prompt = "You are a data visualization expert. You generate syntactically correct Python code."
    user_prompt = f"""Create a {graph_type} using matplotlib for the following data:
    {csv_data}

    Styling requirements:
    - Use a modern, professional color palette (avoid default matplotlib colors)
    - Add grid lines with subtle styling (alpha=0.3)
    - Use clear, readable fonts (fontsize=12 for labels, 14 for title)
    - Add proper titles and axis labels
    - Use tight layout to prevent clipping
    - For multiple series, use distinct colors from a cohesive palette
    - Add legend if multiple data series are present
    - All text on the chart should be of red color

    Return only valid Python code. Include imports. Do not use plt.show()."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000
    )

    print(f"Response: {response}")

    python_code = response.choices[0].message.content
    return python_code


def detect_pdf_from_url(url, file_extension):
    """Detect if a URL points to a PDF file using multiple methods
    
    Args:
        url: The URL to check
        file_extension: The file extension extracted from the URL
        
    Returns:
        bool: True if the URL points to a PDF file, False otherwise
    """
    try:
        import requests
        # Make a HEAD request to check content-type without downloading the full file
        head_response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = head_response.headers.get('content-type', '').lower()
        
        logger.info(f"Content-Type from HEAD request: {content_type}")
        
        # Check if it's a PDF based on content-type
        if 'application/pdf' in content_type or 'pdf' in content_type:
            logger.info("Detected PDF file based on content-type")
            return True
        # Google Drive often returns 'application/octet-stream' for PDFs, so we need to check file magic bytes
        elif content_type == 'application/octet-stream' or 'octet-stream' in content_type:
            logger.info("Content-type is octet-stream, checking file magic bytes...")
            # Download first few bytes to check PDF magic signature
            try:
                headers = {'Range': 'bytes=0-7'}  # Get first 8 bytes
                partial_response = requests.get(url, headers=headers, timeout=10)
                file_header = partial_response.content
                
                # PDF files start with '%PDF' (hex: 25 50 44 46)
                if file_header.startswith(b'%PDF'):
                    logger.info("Detected PDF file based on magic bytes")
                    return True
                else:
                    logger.info(f"File magic bytes: {file_header.hex()[:16]} - not a PDF")
            except Exception as magic_error:
                logger.warning(f"Could not check magic bytes: {str(magic_error)}")
                # Fallback to file extension check
                if file_extension == 'pdf' or '.pdf' in url.lower():
                    logger.info("Detected PDF file based on file extension fallback")
                    return True
        # Also check file extension as fallback
        elif file_extension == 'pdf' or '.pdf' in url.lower():
            logger.info("Detected PDF file based on file extension")
            return True
        else:
            logger.info(f"File detected as non-PDF (content-type: {content_type})")
            return False
                
    except Exception as e:
        logger.warning(f"Could not determine file type via HEAD request: {str(e)}")
        # Fallback to URL-based detection
        url_lower = url.lower()
        is_pdf = ('.pdf' in url_lower or file_extension == 'pdf')
        logger.info(f"Using fallback URL-based detection: is_pdf={is_pdf}")
        return is_pdf


def add_img_base64_snippet(code):
    new_imports = "import io\nimport base64\n"
    code += """
# Save plot to a BytesIO object
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
plt.close()

# Encode image to base64
img_buf.seek(0)
img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
print(img_base64)"""
    return new_imports + code


def upload_file_to_openai(file_content, filename):
    """Upload a file to OpenAI's API and return the file ID"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Enable detailed logging for OpenAI API requests
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        
        # Create a temporary file-like object
        file_obj = io.BytesIO(file_content)
        file_obj.name = filename

        # Upload the file
        response = client.files.create(
            file=file_obj,
            purpose="assistants"
        )

        return response.id, None
    except Exception as e:
        return None, str(e)


def analyze_document_with_vision(file_url, question_title, question_content, approval_criteria):
    """Analyze a document using OpenAI's Chat Completions API with vision input"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Enable detailed logging for OpenAI API requests
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        
        # Log the input parameters for debugging
        logger.info(f"Analyzing document: {file_url}")
        logger.info(f"Question title: {question_title[:50]}..." if len(question_title) > 50 else f"Question title: {question_title}")

        # System prompt for Six Sigma expert
        system_prompt = """You are a Six Sigma expert with extensive experience in process improvement, quality management, and statistical analysis. You specialize in evaluating delegate submissions for Six Sigma projects and certifications.

Analyze the provided document submission thoroughly and provide detailed feedback based on Six Sigma principles.

You must respond with ONLY a valid JSON object in this exact format:
{{
    "analysis": "detailed analysis of the submission",
    "solution": "approved" or "rejected",
    "score": a number between 0 and 10 based on the quality of the submission
}}

Do not include any markdown formatting, code blocks, or additional text. Return only the raw JSON.

For the analysis field, provide comprehensive feedback on the submission.
For the solution field, use exactly "approved" if the submission contains documents that meet the criteria and gives acceptable solution:

If it does not meet these criteria, use exactly "rejected".

The score should reflect the quality:
- 10: Perfect answer meeting all criteria
- 8-9: Strong answer with minimal issues
- 6-7: Meets minimum passing requirements
- 1-5: Does not meet requirements (failing score)"""

        # Construct the prompt
        prompt = f"""QUESTION TITLE:

{question_title}

QUESTION_CONTENT:

{question_content}

APPROVAL_CRITERIA:

{approval_criteria}"""

        # Extract filename from URL and determine file extension
        filename = file_url.split('/')[-1] if '/' in file_url else "document_from_url"
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''

        # Determine if we're dealing with a PDF file
        processed_url = preprocess_google_drive_url(file_url)
        url_lower = processed_url.lower()
        is_pdf = (file_extension == 'pdf' or '.pdf' in url_lower or 
                 'pdf' in url_lower or 'application/pdf' in url_lower)
        
        if is_pdf:
            # Handle PDF processing (both direct PDF files and PDF URLs)
            try:
                import requests
                response_file = requests.get(processed_url)
                file_content = response_file.content

                # Log file content details
                logger.info(f"Downloaded file content size: {len(file_content)} bytes")
                
                # Extract text from PDF
                logger.info("Extracting text from PDF...")
                extracted_text, text_error = extract_text_from_pdf(file_content)
                if text_error:
                    logger.warning(f"Text extraction error: {text_error}")
                else:
                    logger.info(f"Successfully extracted {len(extracted_text)} characters of text")
                
                # Convert PDF to images for visual analysis
                logger.info("Converting PDF to images...")
                pdf_images, conversion_error = convert_pdf_to_images(file_content)
                
                # Log image processing results
                if pdf_images and not conversion_error:
                    hosted_count = sum(1 for ref in pdf_images if ref.startswith('http'))
                    base64_count = len(pdf_images) - hosted_count
                    logger.info(f"✓ PDF converted to {len(pdf_images)} image references")
                    if hosted_count > 0:
                        logger.info(f"  - {hosted_count} images uploaded to temporary hosting service")
                    if base64_count > 0:
                        logger.info(f"  - {base64_count} images converted to base64 data URLs")
                    
                    # Log details of the first image reference for debugging
                    if pdf_images:
                        first_ref = pdf_images[0]
                        if first_ref.startswith('http'):
                            logger.info(f"First image URL: {first_ref[:100]}...")
                        else:
                            logger.info(f"First image is a base64 data URL of length {len(first_ref)} starting with: {first_ref[:50]}...")
                elif conversion_error:
                    logger.error(f"⚠ Image conversion failed: {conversion_error}")

                # Prepare content for analysis
                analysis_content = []
                
                # Add the main prompt
                main_text = f"{prompt}\n\n"
                
                # Include extracted text if available
                if extracted_text and not text_error:
                    main_text += f"EXTRACTED TEXT FROM PDF:\n{extracted_text}\n\n"
                elif text_error:
                    main_text += f"Note: Text extraction failed ({text_error}), analyzing visual content only.\n\n"
                
                # Add visual analysis instruction
                if pdf_images and not conversion_error:
                    main_text += f"Please analyze both the extracted text above and the visual content in the {len(pdf_images)} page(s) below for a comprehensive evaluation."
                    
                    analysis_content.append({
                        "type": "text",
                        "text": main_text
                    })
                    
                    # Add all pages for visual analysis (limit to first 3 pages to avoid token limits)
                    max_pages = min(3, len(pdf_images))
                    for i in range(max_pages):
                        # Handle both regular URLs and base64 data URLs
                        image_ref = pdf_images[i]
                        logger.info(f"Processing image {i+1} of {max_pages}")
                        
                        # Validate image URL before adding to content
                        if image_ref.startswith('http') or image_ref.startswith('data:image/'):
                            # Log the image reference type for debugging
                            if image_ref.startswith('http'):
                                logger.info(f"Using hosted image URL (page {i+1}): {image_ref[:100]}...")
                                
                                # Test URL accessibility
                                try:
                                    import requests
                                    head_response = requests.head(image_ref, timeout=5)
                                    logger.info(f"URL status code: {head_response.status_code}")
                                    if head_response.status_code >= 400:
                                        logger.warning(f"Image URL may not be accessible (status code {head_response.status_code})")
                                except Exception as e:
                                    logger.warning(f"Failed to check image URL: {str(e)}")
                            else:
                                mime_type = image_ref.split(';')[0] if ';' in image_ref else 'unknown'
                                logger.info(f"Using base64 data URL (page {i+1}), MIME type: {mime_type}, length: {len(image_ref)}")
                                
                            # Ensure data URLs are properly formatted for OpenAI
                            if image_ref.startswith('data:'):
                                # Verify it's one of the supported formats
                                supported_formats = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
                                format_found = next((fmt for fmt in supported_formats if fmt in image_ref), None)
                                
                                if not format_found:
                                    logger.warning(f"Unsupported image format in data URL (page {i+1})")
                                    continue
                                else:
                                    logger.info(f"Confirmed supported format: {format_found}")
                            
                            logger.info("Adding image to analysis content")
                            analysis_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_ref
                                }
                            })
                        else:
                            logger.warning(f"Invalid image reference format: {image_ref[:30]}... (skipping)")
                else:
                    main_text += "Please analyze the extracted text above for evaluation."
                    analysis_content.append({
                        "type": "text",
                        "text": main_text
                    })
                
                # If neither text nor images are available, return error
                if (not extracted_text or text_error) and (conversion_error or not pdf_images):
                    return None, f"Failed to process PDF: Text extraction error: {text_error}, Image conversion error: {conversion_error}"

                # Make the API call with combined content
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": analysis_content
                        }
                    ],
                    max_tokens=2000
                )
                
            except Exception as e:
                return None, f"Error processing PDF: {str(e)}"
        else:
            # Use the existing preprocess_google_drive_url function for URL processing
            processed_url = preprocess_google_drive_url(file_url)
            
            # Determine if the URL points to a PDF by checking content-type
            logger.info(f"Checking file type for URL: {processed_url}")
            is_pdf_url = False
            
            is_pdf_url = detect_pdf_from_url(processed_url, file_extension)
            
            if is_pdf_url:
                # Handle PDF URLs by downloading and processing for both text and visual analysis
                try:
                    import requests
                    response_file = requests.get(processed_url)
                    file_content = response_file.content

                    # Extract text from PDF
                    extracted_text, text_error = extract_text_from_pdf(file_content)
                    
                    # Convert PDF to images for visual analysis
                    pdf_images, conversion_error = convert_pdf_to_images(file_content)
                    
                    # Log image processing results
                    if pdf_images and not conversion_error:
                        hosted_count = sum(1 for ref in pdf_images if ref.startswith('http'))
                        base64_count = len(pdf_images) - hosted_count
                        print(f"✓ PDF converted to {len(pdf_images)} image references")
                        if hosted_count > 0:
                            print(f"  - {hosted_count} images uploaded to temporary hosting service")
                        if base64_count > 0:
                            print(f"  - {base64_count} images converted to base64 data URLs")
                    elif conversion_error:
                        print(f"⚠ Image conversion failed: {conversion_error}")

                    # Prepare content for analysis
                    analysis_content = []
                    
                    # Add the main prompt
                    main_text = f"{prompt}\n\n"
                    
                    # Include extracted text if available
                    if extracted_text and not text_error:
                        main_text += f"EXTRACTED TEXT FROM PDF:\n{extracted_text}\n\n"
                    elif text_error:
                        main_text += f"Note: Text extraction failed ({text_error}), analyzing visual content only.\n\n"
                    
                    # Add visual analysis instruction
                    if pdf_images and not conversion_error:
                        main_text += f"Please analyze both the extracted text above and the visual content in the {len(pdf_images)} page(s) below for a comprehensive evaluation."
                        
                        analysis_content.append({
                            "type": "text",
                            "text": main_text
                        })
                        
                        # Add all pages for visual analysis (limit to first 3 pages to avoid token limits)
                        max_pages = min(3, len(pdf_images))
                        for i in range(max_pages):
                            # Use hosted image URL directly for OpenAI Vision API
                            hosted_url = pdf_images[i]
                            analysis_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": hosted_url
                                }
                            })
                    else:
                        main_text += "Please analyze the extracted text above for evaluation."
                        analysis_content.append({
                            "type": "text",
                            "text": main_text
                        })
                    
                    # Make the API call with combined content
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": analysis_content
                            }
                        ],
                        max_tokens=2000
                    )
                    
                except Exception as e:
                    return None, f"Error processing PDF from URL: {str(e)}"
            else:
                # Make the API call for image files using the processed URL
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": processed_url
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000
                )

        # Parse the JSON response
        try:
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)
            
            # Return structured response
            return {
                "success": True,
                "analysis": parsed_response.get("analysis", "No analysis provided"),
                "score": parsed_response.get("score", 0),
                "solution": parsed_response.get("solution", "rejected")
            }, None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None, f"Invalid JSON response from AI: {str(e)}"
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return None, f"Error processing AI response: {str(e)}"
    except Exception as e:
        return None, str(e)


def mark_answer(question_title, question_content, approval_criteria, text_submission):
    """
    Grade answers using OpenAI API based on approval criteria.

    Args:
        question_title (str): The title of the question
        question_content (str): The content/description of the question
        approval_criteria (str): The criteria for approval
        text_submission (str): The text submission to be graded

    Returns:
        dict: Response containing success status, solution, score, and feedback
    """
    try:
        # Validate required parameters
        if not all([question_title, question_content, approval_criteria, text_submission]):
            return {
                'success': False,
                'error': 'Invalid input data. Please check your request.',
                'validation_errors': 'All fields (question_title, question_content, approval_criteria, text_submission) are required'
            }

        # System prompt with score requirement
        system_prompt = f"""You are a six sigma expert grading answers. You'll be given a question title, question content, a text submission, and an approval criteria.

You must respond with ONLY a valid JSON object in this exact format:
{{
    "solution": "Approved" or detailed feedback,
    "score": a number between 0 and 10 based on the quality of the submission
}}

Do not include any markdown formatting, code blocks, or additional text. Return only the raw JSON.

If the submission meets the criteria, use exactly "Approved" as the solution.
If not, provide constructive feedback speaking directly to the user.

The score should reflect the quality:
- 10: Perfect answer meeting all criteria
- 8-9: Strong answer with minimal issues
- 6-7: Meets minimum passing requirements
- 1-5: Does not meet requirements (failing score)

## Approval criteria: {approval_criteria}"""

        user_prompt = f"""## Question Title: {question_title}

## Question Content: {question_content}

## Text Submission: {text_submission}"""

        # Make API request to OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        # Decode the JSON response with error handling
        try:
            decoded_response = json.loads(content)
            solution = 'approved' if decoded_response['solution'] == 'Approved' else 'rejected'
            score = float(decoded_response['score'])
            feedback = decoded_response['solution']
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback for backward compatibility or error cases
            solution = 'approved' if content == 'Approved' else 'rejected'
            score = 0.0  # Default score when parsing fails
            feedback = content

        return {
            'success': True,
            'solution': solution,
            'score': score,
            'response': feedback if feedback else 'No response generated'
        }

    except requests.exceptions.RequestException as e:
        print(f'API Request failed: {str(e)}')
        return {
            'success': False,
            'error': 'Failed to communicate with the AI service. Please try again later.'
        }
    except json.JSONDecodeError as e:
        print(f'JSON parsing failed: {str(e)}')
        return {
            'success': False,
            'error': 'Failed to process the AI response. Please try again.'
        }
    except Exception as e:
        print(f'Unexpected error: {str(e)}')
        return {
            'success': False,
            'error': 'An unexpected error occurred. Please try again later.'
        }


def execute_python_code(code):
    try:
        # Remove markdown formatting if present - handle all variations
        code = code.strip()

        # Handle various markdown code block formats
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```py"):
            code = code[5:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        # Clean up any remaining whitespace
        code = code.strip()

        # Basic syntax validation before execution
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as syntax_err:
            return None, f"Syntax error in generated code: {str(syntax_err)}. Please try again with a different request."

        # Add the base64 image generation snippet
        code = add_img_base64_snippet(code)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(code.encode())
            temp_file_path = temp_file.name

        python_executable = shutil.which("python3") or shutil.which("python")

        # TODO: remove
        if not python_executable:
            raise RuntimeError("Python interpreter not found")
        # Execute the generated Python code
        result = subprocess.run(["/home/sigmafy_usr/venv/bin/python3", temp_file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip(), None
        else:
            return None, result.stderr
    except Exception as e:
        return None, str(e)
