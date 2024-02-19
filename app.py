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

def generate_report(model_name, supported_ops, op_counts):
    supported_ops = [['Abs', 1], ['Add', 7], ['And', 1], ['ArgMax', 1], ['ArgMin', 1]]
    op_counts = {"Abs": 150, "Add": 14, "MatMul": 3}
    onnx_version = float("inf") # not implemented yet
    # Convert supported_ops to a dict for easier access
    supported_ops_dict = {op[0]: op[1] for op in supported_ops}

    # Prepare lists to store supported and unsupported ops
    supported = []
    unsupported = []

    # Check each op in op_appearance_counts against supported_ops
    for op, count in op_counts.items():
        if op in supported_ops_dict and onnx_version >= supported_ops_dict[op]:
            supported.append((op, count, supported_ops_dict[op]))
        else:
            unsupported.append((op, count, supported_ops_dict.get(op, 'N/A')))

    # Sort ops first by support status, then by occurrence
    # Note: Python's sort is stable, so we can sort by count first, then by support status
    supported.sort(key=lambda x: x[1], reverse=True)
    unsupported.sort(key=lambda x: x[1], reverse=True)

    # Combine and sort the final list so unsupported ops come first
    ops = unsupported + supported

    # Generate HTML content
    html_content = f"<html><head><title>ONNX Operations Report for {model_name}</title></head><body>"
    html_content += f"<h1>ONNX Operations Report for {model_name}</h1>"
    html_content += "<table border='1'><tr><th>Operation</th><th>Count</th><th>Supported Since Opset</th><th>Status</th></tr>"
    for op, count, since_version in ops:
        color = 'red' if op in [u[0] for u in unsupported] else 'black'
        status = 'Unsupported' if color == 'red' else 'Supported'
        html_content += f"<tr style='color:{color};'><td>{op}</td><td>{count}</td><td>{since_version}</td><td>{status}</td></tr>"
    html_content += "</table>"
    html_content += "</body></html>"

    return html_content

# The following line is used to run the server with uvicorn from command line:
# uvicorn filename:app --reload
# Replace `filename` with the name of your Python script.

from ops_torch_mlir_onnx import get_supported_ops
from fastapi.responses import HTMLResponse

@app.get("{model_name:path}")
async def get_op_counts(model_name: str, response_class=HTMLResponse):
    if model_name.startswith("/"):
        model_name = model_name[1:]
    try:
        op_counts = whatops(model_name)
        supported_ops = get_supported_ops()
        # return result of generate report
        return HTMLResponse(
            generate_report(model_name,supported_ops, op_counts))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

