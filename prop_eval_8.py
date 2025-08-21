import numpy as np
import onnx
import re
import os
from src.vnnlib import getIoNodes
from src.util import predictWithOnnxruntime, removeUnusedInitializers

def parse_input_file(filename):
    """Parse input vectors from a text file."""
    inputs = []
    with open(filename, 'r') as file:
        content = file.read()
    pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
    matches = re.findall(pattern, content)
    for match in matches:
        input_values = [float(val) for val in match]
        inputs.append(input_values)
    return inputs

def onnx_evaluate(onnx_model, input_vector, input_dtype, input_shape):
    """Runs a single input vector through the ONNX model and returns the output."""
    inputs = np.array(input_vector, dtype=input_dtype).reshape(input_shape)
    output = predictWithOnnxruntime(onnx_model, inputs)
    return output.flatten()

def check_property_8_violation_cause(output_vector):
    """
    Checks the output vector to determine the cause of the Property 8 violation.

    Args:
        output_vector (np.array): The output from the neural network (Y_0 to Y_4).

    Returns:
        str: "WL" if Weak Left is minimal, "COC" if Clear of Conflict is minimal, 
             "Unknown" otherwise.
    """
    Y_0, Y_1, Y_2, Y_3, Y_4 = output_vector

    # Property 8 Logic from VNNLIB file:
    # (or (and (<= Y_2 Y_0) (<= Y_2 Y_1))  ; WL is minimal
    #     (and (<= Y_3 Y_0) (<= Y_3 Y_1))  ; COC is minimal 
    #     (and (<= Y_4 Y_0) (<= Y_4 Y_1))  ; COC is minimal 
    # )
    
    is_wl_minimal = (Y_1 >= Y_0) and (Y_1 >= Y_2) and (Y_1 >= Y_3) and (Y_1 >= Y_4)
    is_coc_minimal = ((Y_0 >= Y_1) and (Y_0 >= Y_2)) or ((Y_0 >= Y_3) and (Y_0 >= Y_4))

    if is_wl_minimal:
        return "WL"
    elif is_coc_minimal:
        return "COC"
    else:
        # This case should not happen for a valid adversarial input for Prop 8
        return "Unknown"


adversarial_filename = "adv_merged.txt"
onnx_filename = "benchmarks/acasxu/ACASXU_run2a_2_9_batch_2000.onnx"

# Load data
print(f"Loading adversarial data from '{adversarial_filename}'")
adversarial_inputs = parse_input_file(adversarial_filename)

if not adversarial_inputs:
    print("No adversarial inputs found. Exiting.")
    exit()
print(f"Successfully loaded {len(adversarial_inputs)} adversarial inputs.")

# Load ONNX
print(f"\nLoading ONNX model from '{onnx_filename}'")
if not os.path.exists(onnx_filename):
    print(f"Error: ONNX model not found at '{onnx_filename}'.")
    exit()
    
onnx_model = onnx.load(onnx_filename)
onnx_model = removeUnusedInitializers(onnx_model)
inp, _, inp_dtype = getIoNodes(onnx_model)
inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
print("ONNX model loaded successfully.")

wl_count = 0
coc_count = 0
unknown_count = 0

for adv_input in adversarial_inputs:
    # Get the model's output for the current input
    output_scores = onnx_evaluate(onnx_model, adv_input, inp_dtype, inp_shape)
    
    # Determine the cause of the violation
    cause = check_property_8_violation_cause(output_scores)
    
    if cause == "WL":
        wl_count += 1
    elif cause == "COC":
        coc_count += 1
    else:
        unknown_count += 1

#Results
total_inputs = len(adversarial_inputs)
print(f"Total Inputs Analyzed: {total_inputs}")

if total_inputs > 0:
    wl_percentage = (wl_count / total_inputs) * 100
    coc_percentage = (coc_count / total_inputs) * 100
    
    print(f"Inputs caused by Minimal WL: {wl_count} ({wl_percentage:.2f}%)")
    print(f"Inputs caused by Minimal COC: {coc_count} ({coc_percentage:.2f}%)")
    
    if unknown_count > 0:
        print(f"Warning: {unknown_count} inputs did not match WL or COC minimal conditions.")
