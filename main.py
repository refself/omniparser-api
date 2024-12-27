from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import io
from PIL import Image
import torch
import numpy as np
import os

# Existing imports
import numpy as np
import torch
from PIL import Image
import io

from utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)
import torch

from ultralytics import YOLO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
try:
    yolo_model = YOLO("weights/icon_detect_v1_5/model_v1_5.pt").to(DEVICE)
except:
    yolo_model = YOLO("weights/icon_detect_v1_5/model_v1_5.pt")

# Initialize Florence model
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(DEVICE)
except:
    model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
caption_model_processor = {"processor": processor, "model": model}
print("finish loading model!!!")

app = FastAPI(title="OmniParser API")

class ProcessRequest(BaseModel):
    box_threshold: float = 0.05
    iou_threshold: float = 0.1
    use_paddleocr: bool = False
    imgsz: int = 1920
    icon_process_batch_size: int = 64

class ProcessResponse(BaseModel):
    image: str  # Base64 encoded image
    parsed_content_list: str
    label_coordinates: str

def process(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool = False,
    imgsz: int = 1920,
    icon_process_batch_size: int = 64
) -> ProcessResponse:
    image_save_path = "imgs/saved_image_demo.png"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
    
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
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
    print("finish processing")
    
    # Format parsed_content_list similar to Gradio demo
    parsed_content_list_str = '\n'.join([
        f'type: {x["type"]}, content: {x["content"]}, interactivity: {x["interactivity"]}'
        for x in parsed_content_list
    ])

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Clean up temporary file
    if os.path.exists(image_save_path):
        os.remove(image_save_path)

    return ProcessResponse(
        image=img_str,
        parsed_content_list=parsed_content_list_str,
        label_coordinates=str(label_coordinates),
    )

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
    - box_threshold: Threshold for removing bounding boxes with low confidence (0.01-1.0)
    - iou_threshold: Threshold for removing bounding boxes with large overlap (0.01-1.0)
    - use_paddleocr: Whether to use PaddleOCR instead of EasyOCR
    - imgsz: Icon detection image size (640-3200)
    - icon_process_batch_size: Batch size for icon processing (1-256)
    """
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        response = process(
            image_input,
            box_threshold,
            iou_threshold,
            use_paddleocr,
            imgsz,
            icon_process_batch_size
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    """Welcome endpoint with basic API information"""
    return {
        "message": "Welcome to OmniParser API",
        "description": "A screen parsing tool to convert general GUI screen to structured elements",
        "documentation": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
