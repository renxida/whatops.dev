import onnx
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
from collections import Counter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


def export_model_to_onnx(model_name, onnx_file_path):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name)

    # Initialize the model
    model = AutoModel.from_config(config)

    # try to get dummy input from tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dummy_input = tokenizer("Hello, my dog is cute", return_tensors="pt")[
            "input_ids"
        ]
    except:
        print(
            f"Failed to get dummy input for model {model_name} via tokenizer. Checking resnet..."
        )

        # check if model is resnet
        if "resnet" in model_name:
            print("Model is resnet")
            # Load the image transformer
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            # generate random image
            image = Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))
            # Apply the transformations
            input_tensor = transform(image)
            # Add the batch dimension
            input_batch = input_tensor.unsqueeze(0)
            # dummy input
            dummy_input = input_batch
        else:
            print("Model is not resnet. Trying random input of shape (1, 1, 768)...")
            dummy_input = torch.randn(1, 1, 768)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def analyze_onnx_model(onnx_file_path):
    # Load the ONNX model
    model = onnx.load(onnx_file_path)

    # Counter for operations
    op_counter = Counter()

    # Iterate through the nodes in the ONNX graph
    for node in model.graph.node:
        op_counter[node.op_type] += 1

    return op_counter


def format_op_counts(op_counts):
    return "\n".join([f"{op}: {count}" for op, count in op_counts.items()])


def whatops(model_name):
    if model_name.startswith("/"):
        model_name = model_name[1:]
    onnx_file_path = f"{model_name.replace('/', '_')}.onnx"
    # export if not exist
    if not os.path.exists(onnx_file_path):
        export_model_to_onnx(model_name, onnx_file_path)
    op_counts = analyze_onnx_model(onnx_file_path)
    return op_counts


if __name__ == "__main__":
    # model_name = "microsoft/resnet-50"  # Example model
    model_name = "bert-base-uncased"  # Example model
    op_counts = whatops(model_name)
    print(format_op_counts(op_counts))
