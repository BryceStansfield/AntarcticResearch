from paddleocr import PaddleOCR
import pymupdf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import pathlib
import multiprocessing
import sys
import os

def parse_pdf(pdf_path):
    model = PaddleOCR(enable_mkldnn=False, ocr_version='PP-OCRv4', 
                      text_det_box_thresh=0.05,
                      text_det_thresh=0.025)

    text_pieces = []

    doc = pymupdf.open(pdf_path)
    for page_num, page in enumerate(doc):
        embedded_text = page.get_text()
        if len(embedded_text) > 50:
            text_pieces.append(embedded_text)
            continue

        # Render page to a pixmap (surface image)
        pix = page.get_pixmap(dpi=300)
        im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        results = model.predict(im)

        for result in results:
            text_pieces += result['rec_texts']
            image = Image.fromarray(im, mode="RGB")
            draw = ImageDraw.Draw(image)

            for line in result["rec_polys"]:
                draw.polygon([list(p) for p in line], outline='red', width=5)
            
    
    full_text = ' '.join(text_pieces)
    return full_text

def parse_pdf_and_save(pdf_path: pathlib.Path, force_rerun = False):
    full_text_path = pdf_path.with_suffix(f"{pdf_path.suffix}.ocr.txt")

    if full_text_path.exists() and not force_rerun:
        return
    
    full_text = parse_pdf(pdf_path)
    with open(full_text_path, "w") as f:
        f.write(full_text)
    return

def no_logging():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)

def ocr_full_directory(directory_path: pathlib.Path):
    children = [item for item in directory_path.iterdir()]
    
    children_to_parse = []
    for child in children:
        if child.suffix == ".pdf":
            children_to_parse.append(child)
    
    with multiprocessing.Pool(processes=8, initializer=no_logging) as pool:
        pool.map(parse_pdf_and_save, children_to_parse)

if __name__ == "__main__":
    ocr_full_directory(pathlib.Path("/home/brycestansfield/Documents/Master_DS/ResearchWithMichael/AntarcticResearch/data/final_reports"))