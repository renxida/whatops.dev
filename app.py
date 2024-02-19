from fastapi import FastAPI, HTTPException
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
from collections import Counter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import onnx

app = FastAPI()

from whatops import whatops

@app.get("{model_name:path}")
async def get_op_counts(model_name: str):
    try:
        op_counts = whatops(model_name)
        return op_counts
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))