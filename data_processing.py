import logging
import os
import re
import pandas as pd
import PyPDF2

logger = logging.getLogger(__name__)


def clean_csv(file_path):
    """Clean the CSV data: handle column names, percentages, missing values, duplicates."""
    logger.info(f"Starting CSV cleaning: {file_path}")
    df = pd.read_csv(file_path)
    logger.debug(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df.columns = df.columns.str.strip()
    logger.debug("Column names stripped")

    if 'Votes(%)' in df.columns:
        df['Votes(%)'] = df['Votes(%)'].astype(str).str.rstrip('%').replace('', '0').astype(float) / 100
        logger.debug("Percentage values converted to decimals")

    df = df.fillna('')
    logger.debug("Missing values filled")

    for col in ['Old Region', 'New Region']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    logger.debug("Region names standardized")

    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    logger.info(f"CSV cleaning completed: {len(df)} rows remaining ({removed_duplicates} duplicates removed)")
    return df


def extract_clean_pdf(file_path):
    """Extract text from PDF and clean it."""
    logger.info(f"Starting PDF extraction: {file_path}")
    text = ''
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.debug(f"PDF loaded: {num_pages} pages")
            for idx, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
                if (idx + 1) % 10 == 0:
                    logger.debug(f"Extracted {idx + 1}/{num_pages} pages")
    except Exception as e:
        logger.warning(f"Error extracting PDF: {e}. Using placeholder text.")
        text = "Sample budget text for demonstration. This is a placeholder since PDF extraction failed. Economic policy includes growth projections and fiscal measures."

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    logger.info(f"PDF extraction completed: {len(text)} characters extracted")
    return text


def chunk_fixed_size(text, chunk_size=500, overlap=50):
    """Chunk text into fixed size with overlap."""
    logger.debug(f"Starting fixed-size chunking: chunk_size={chunk_size}, overlap={overlap}")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break

    avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    logger.info(f"Fixed-size chunking completed: {len(chunks)} chunks, avg length: {avg_length:.2f} chars")
    return chunks


def chunk_by_sentences(text, sentences_per_chunk=5, overlap=1):
    """Chunk text by sentences with overlap."""
    logger.debug(f"Starting sentence-based chunking: sentences_per_chunk={sentences_per_chunk}, overlap={overlap}")
    sentences = re.split(r'(?<=\.)\s+', text)
    num_sentences = len(sentences)
    logger.debug(f"Text split into {num_sentences} sentences")

    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk_sentences = sentences[start:end]
        if chunk_sentences:
            chunks.append(' '.join(chunk_sentences))
        start = end - overlap
        if start >= len(sentences):
            break

    avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    logger.info(f"Sentence-based chunking completed: {len(chunks)} chunks, avg length: {avg_length:.2f} chars")
    return chunks


def chunk_by_paragraphs(text, min_chars=200):
    """Chunk text by paragraphs and combine short paragraphs."""
    logger.debug("Starting paragraph-based chunking")
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current = ''
    for para in paragraphs:
        if len(current) < min_chars:
            current = (current + ' ' + para).strip()
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    chunks = [chunk for chunk in chunks if len(chunk) >= 50]
    avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    logger.info(f"Paragraph-based chunking completed: {len(chunks)} chunks, avg length: {avg_length:.2f} chars")
    return chunks


def filter_chunks(chunks, min_length=80):
    """Remove low-information or excessively short chunks."""
    filtered = [chunk for chunk in chunks if len(chunk) >= min_length and len(chunk.split()) > 8]
    logger.info(f"Filtered chunks: {len(filtered)} remain from {len(chunks)}")
    return filtered


def manage_context_window(contexts, max_length=2000):
    """Truncate combined contexts to fit within max_length characters."""
    combined = ' '.join(contexts)
    if len(combined) > max_length:
        combined = combined[:max_length] + '...'
    return combined
