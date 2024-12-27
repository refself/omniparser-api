from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import io
from PIL import Image
import torch
import numpy as np
import os
import argparse
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process model paths and names.')
parser.add_argument('--icon_detect_model', 
                   type=str, 
                   default='weights/icon_detect_v1_5/model_v1_5.pt',
                   help='Path to the YOLO model weights')
parser.add_argument('--icon_caption_model', 
                   type=str, 
                   default='florence2',
                   help='Name of the caption model')

args = parser.parse_args()

# Initialize models
print(f"Loading YOLO model from: {args.icon_detect_model}")
yolo_model = get_yolo_model(model_path=args.icon_detect_model)

print(f"Loading caption model: {args.icon_caption_model}")
if args.icon_caption_model == 'florence2':
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence"
    )
elif args.icon_caption_model == 'blip2':
    caption_model_processor = get_caption_model_processor(
        model_name="blip2",
        model_name_or_path="weights/icon_caption_blip2"
    )

app = FastAPI(title="OmniParser API", description=MARKDOWN)

class ProcessResponse(BaseModel):
    image: str  # Base64 encoded image
    parsed_content_list: str

def process(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
    icon_process_batch_size: int,
) -> tuple[Image.Image, str]:
    image_save_path = 'imgs/saved_image_demo.png'
    os.makedirs('imgs', exist_ok=True)
    
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        batch_size=icon_process_batch_size
    )

    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    
    parsed_content_list = '\n'.join([
        f'type: {x["type"]}, content: {x["content"]}, interactivity: {x["interactivity"]}'
        for x in parsed_content_list
    ])
    
    # Cleanup
    if os.path.exists(image_save_path):
        os.remove(image_save_path)
        
    return image, parsed_content_list

@app.post("/process_image", response_model=ProcessResponse)
async def process_image(
    image_file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = False,
    imgsz: int = 1920,
    icon_process_batch_size: int = 64
):
    """
    Process an image using OmniParser.
    
    Parameters:
    - image_file: The input image file
    - box_threshold: Confidence threshold for bounding boxes (0.01-1.0)
    - iou_threshold: IOU threshold for overlap detection (0.01-1.0)
    - use_paddleocr: Whether to use PaddleOCR instead of EasyOCR
    - imgsz: Icon detection image size (640-3200)
    - icon_process_batch_size: Batch size for processing (1-256)
    """
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        processed_image, parsed_content = process(
            image_input,
            box_threshold,
            iou_threshold,
            use_paddleocr,
            imgsz,
            icon_process_batch_size
        )
        
        # Convert processed image to base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return ProcessResponse(
            image=img_str,
            parsed_content_list=parsed_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    """Welcome endpoint with API information"""
    return {
        "title": "OmniParser API",
        "description": MARKDOWN,
        "model_info": {
            "icon_detect_model": args.icon_detect_model,
            "icon_caption_model": args.icon_caption_model
        },
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server with models:")
    print(f"- Icon Detection: {args.icon_detect_model}")
    print(f"- Caption Model: {args.icon_caption_model}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
