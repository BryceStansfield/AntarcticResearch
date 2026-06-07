from paddleocr import PaddleOCR
import pymupdf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import pathlib


def parse_pdf(pdf_path):
    model = PaddleOCR(enable_mkldnn=False, ocr_version='PP-OCRv4', 
                      det_db_box_thresh=0.1,
                      det_db_thresh=0.05)

    text_pieces = []

    doc = pymupdf.open(pdf_path)
    for page_num, page in enumerate(doc):
        # Render page to a pixmap (surface image)
        pix = page.get_pixmap(dpi=300)
        im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        results = model.predict(im)

        for result in results:
            text_pieces += result['rec_texts']
            print(result['rec_texts'])
            image = Image.fromarray(im, mode="RGB")
            draw = ImageDraw.Draw(image)
            print(result)

            for line in result["rec_polys"]:
                draw.polygon([list(p) for p in line], outline='red', width=5)
            
            image.save(f"debug/Debug_{page_num}.png")
    
    full_text = ' '.join(text_pieces)
    return full_text

def parse_pdf_and_save(pdf_path: pathlib.Path, force_rerun = False):
    full_text_path = pdf_path / ".ocr.txt"

    if full_text_path.exists() and not force_rerun:
        return
    
    full_text = parse_pdf(pdf_path)
    with open(full_text_path, "w") as f:
        f.write(full_text)
    return
    

print(parse_pdf("/home/brycestansfield/Documents/Master_DS/ResearchWithMichael/AntarcticResearch/data/final_reports/ATCM47_WW014_e.pdf"))