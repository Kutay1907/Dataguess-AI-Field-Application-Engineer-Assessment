from ultralytics import YOLO
import torch
import numpy as np
import onnx
import onnxruntime as ort
import os
import time

try:
    import onnxsim
except ImportError:
    onnxsim = None
    print("Warning: onnxsim not installed, skipping simplification.")

def export_model(model_path="models/latest.pt", output_path="models/model.onnx"):
    """
    Exports YOLOv8 model to ONNX with dynamic axes and validation.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} does not exist. Train first.")

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # 2. Export to ONNX
    # Opset 12 is chosen for broad compatibility (TensorRT, mobile). 
    # Opset 17 is available but 12 fits most standard deployment needs.
    print("Exporting to ONNX...")
    path = model.export(
        format="onnx", 
        dynamic=True, 
        opset=12,
        simplify=False # we use onnxsim manually for better control
    )
    
    # 3. Simplify (Optional but recommended)
    if onnxsim:
        print("Simplifying with onnxsim...")
        onnx_model = onnx.load(path)
        model_simp, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_path)
            print(f"Simplified model saved to {output_path}")
        else:
            print("Simplification check failed, using original.")
            os.rename(path, output_path)
    else:
        os.rename(path, output_path)

    print(f"Model exported to {output_path}")
    return output_path

def validate_onnx(onnx_path, torch_model_path):
    """
    Validates ONNX model against PyTorch model.
    """
    print("\nStarting validation...")
    
    # Prepare dummy input (batch=1, ch=3, h=640, w=640)
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # 1. PyTorch Output
    torch_model = YOLO(torch_model_path)
    # Access internal pytorch model for raw output
    torch_model.model.eval()
    with torch.no_grad():
        raw_torch_output = torch_model.model(torch.from_numpy(dummy_input))
        # Handle list/tuple output (sometimes happens with different heads)
        if isinstance(raw_torch_output, (list, tuple)):
            torch_out_np = raw_torch_output[0].numpy()
        else:
            torch_out_np = raw_torch_output.numpy()

    # 2. ONNX Runtime Output (with Optimizations)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    
    ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    start_time = time.perf_counter()
    outputs_onnx = ort_session.run([output_name], {input_name: dummy_input})
    end_time = time.perf_counter()
    
    output_onnx_tensor = outputs_onnx[0]

    # 3. Compare
    print(f"PyTorch Output Shape: {torch_out_np.shape}")
    print(f"ONNX    Output Shape: {output_onnx_tensor.shape}")
    print(f"Inference Time (ONNX): {(end_time - start_time)*1000:.2f} ms")

    # Error Metrics
    # Higher tolerance due to fp32/fp16 differences or optimization
    try:
        np.testing.assert_allclose(torch_out_np, output_onnx_tensor, rtol=1e-3, atol=1e-4)
        print("✅ Validation PASSED: ONNX outputs match PyTorch outputs.")
    except AssertionError as e:
        print(f"❌ Validation FAILED: {e}")
        # Print stats
        diff = np.abs(torch_out_np - output_onnx_tensor)
        print(f"Max diff: {np.max(diff)}")
        print(f"Mean diff: {np.mean(diff)}")

if __name__ == "__main__":
    final_onnx = export_model()
    validate_onnx(final_onnx, "models/latest.pt")
