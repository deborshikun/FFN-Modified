import numpy as np
import time
import random
import numpy
import onnx
import onnxruntime as rt
from random import randint
import sys
sys.path.append('..')

from src.vnnlib import readVnnlib, getIoNodes
from src.util import predictWithOnnxruntime, removeUnusedInitializers
from src.util import  findObjectiveFuncionType, checkAndSegregateSamplesForMaximization, checkAndSegregateSamplesForMinimization

'Number of times FFN runs'
numRuns=100

'Number of samples in each FFN run'
numSamples=150

supersats = {}
superunsats = {}

# Global set to track already printed adversarial inputs to avoid duplicates
printed_adversarial_inputs = set()

def reset_adversarial_tracking():
    """Reset the global tracking set for adversarial inputs. 
    Call this at the beginning of each new instance evaluation."""
    global printed_adversarial_inputs
    printed_adversarial_inputs.clear()


def onnxEval(onnxModel,inVals,inpDtype, inpShape):
   flattenOrder='C'
   inputs = np.array(inVals, dtype=inpDtype)
   inputs = inputs.reshape(inpShape, order=flattenOrder) # check if reshape order is correct
   assert inputs.shape == inpShape

   output = predictWithOnnxruntime(onnxModel, inputs)
   flatOut = output.flatten(flattenOrder) # check order, 'C' for row major order
   return flatOut

# --- MODIFIED FUNCTION ---
def propCheck(inputs, specs, outputs):
    found_adv = False
    adv_inputs = []
    non_adv_inputs = [] # New list for non-adversarial inputs
    i = len(supersats)
    k = len(superunsats)
    
    # Check for each property in specs
    for propMat, propRhs in specs:
        vec = propMat.dot(outputs)
        sat = np.all(vec <= propRhs)
        if sat:
            supersats[i] = inputs
            found_adv = True
            adv_inputs.append(inputs)
            
            # Convert inputs to tuple so it can be added to set (lists are not hashable)
            inputs_tuple = tuple(inputs)
            
            # Only print if this adversarial input hasn't been printed before
            if inputs_tuple not in printed_adversarial_inputs:
                print(f"Adversarial input found: {inputs}")
                printed_adversarial_inputs.add(inputs_tuple)
            
            i += 1
        else:
            # This is a non-adversarial input
            superunsats[k] = inputs
            non_adv_inputs.append(inputs) # Add to our new list
            k += 1

    # Return all three results
    return found_adv, adv_inputs, non_adv_inputs
    

def learning(cpos,cneg,iRange,numInputs):
    for i in range (len(cneg)):
        nodeSelect = randint(0,int(numInputs)-1)
        cp=cpos[0][0]
        cn=cneg[i][0]
        cposInVal=cp[nodeSelect]
        cnegInVal=cn[nodeSelect]
        if( cposInVal > cnegInVal):
            temp = round(random.uniform(cnegInVal, cposInVal), 6)
            if ( temp <= iRange[nodeSelect][1] and temp >= iRange[nodeSelect][0]):
                iRange[nodeSelect][0]=temp
        else:
            if (cposInVal < cnegInVal):
                temp = round(random.uniform(cposInVal, cnegInVal), 6)
                if ( temp <= iRange[nodeSelect][1] and temp >= iRange[nodeSelect][0]):
                   iRange[nodeSelect][1]=temp


# --- MODIFIED FUNCTION ---
# I continue sampling until I find an adversarial input or exhaust the sampling attempts
def makeSample(onnxModel, numInputs, inRanges, samples, specs, inpDtype, inpShape):
    sampleInputList = []
    all_adv_inputs = []
    all_non_adv_inputs = [] # New list for non-adversarial inputs
    
    # Generates all samples and stores in a list
    for k in range(numSamples):
        j = 0
        while (j < 5):
            inValues = []
            for i in range(numInputs):
                inValues.append(round(random.uniform(inRanges[i][0], inRanges[i][1]), 6))

            if (inValues in sampleInputList):
                j = j + 1
            else:
                break
        sampleInputList.append(inValues)

        # onnx model evaluation
        sampleVal = onnxEval(onnxModel, inValues, inpDtype, inpShape)
        
        # checking property with onnx evaluation outputs
        found_adv, adv_inputs, non_adv_inputs = propCheck(inValues, specs, sampleVal) # Capture non_adv_inputs
        
        if found_adv:
            all_adv_inputs.extend(adv_inputs)
        
        # Collect non-adversarial inputs
        if non_adv_inputs:
            all_non_adv_inputs.extend(non_adv_inputs)
        
        s = []
        s.append(inValues)
        s.append(sampleVal)
        samples.append(s)
    
    # Return both lists
    return len(all_adv_inputs) > 0, all_adv_inputs, all_non_adv_inputs

# --- MODIFIED FUNCTION ---
def runSample(onnxModel, numInputs, numOutputs, inputRange, tAndOT, spec, inpDtype, inpShape):
    oldPosSamples = []
    target = tAndOT[0]
    objectiveType = tAndOT[1]
    all_advinps = []
    all_non_advinps = [] # New list for non-adversarial inputs
    found_adv = False

    # Run FFN for numRuns
    for k in range(numRuns):
        samples = []
        posSamples = []
        negSamples = []

        ret, advinps, non_advinps = makeSample(onnxModel, numInputs, inputRange, samples, spec, inpDtype, inpShape)
        
        if ret and advinps:
            found_adv = True
            # Only add adversarial inputs that are not already in our collection
            for adv_inp in advinps:
                if adv_inp not in all_advinps:
                    all_advinps.append(adv_inp)
        
        # Collect non-adversarial inputs from the sample run
        if non_advinps:
            # Only add non-adversarial inputs that are not already in our collection
            for non_adv_inp in non_advinps:
                if non_adv_inp not in all_non_advinps:
                    all_non_advinps.append(non_adv_inp)

        # Segregate sample list into positive and negative samples
        if (objectiveType == 1):
           checkAndSegregateSamplesForMinimization(posSamples, negSamples, samples, oldPosSamples, target)
        else:
           checkAndSegregateSamplesForMaximization(posSamples, negSamples, samples, oldPosSamples, target)
        oldPosSamples = posSamples

        # Check if further sampling is possible
        flag = False
        for i in range(numInputs):
            if (inputRange[i][1] - inputRange[i][0] > 0.000001):
               flag = True
               break

        if (flag == False):
           return ("unknown" if not found_adv else "violated"), all_advinps, all_non_advinps

        learning(posSamples, negSamples, inputRange, numInputs)
    
    return ("timeout" if not found_adv else "violated"), all_advinps, all_non_advinps

# --- MODIFIED FUNCTION ---
#SampleEval function
def sampleEval(onnxFilename, vnnlibFilename):
    # Reset adversarial tracking for each new instance
    reset_adversarial_tracking()
    
    onnxModel = onnx.load(onnxFilename)
    onnx.checker.check_model(onnxModel, full_check=True)
    onnxModel = removeUnusedInitializers(onnxModel)
    inp, out, inpDtype = getIoNodes(onnxModel)
    inpShape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    outShape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    numInputs = 1
    numOutputs = 1
    for n in inpShape:
        numInputs *= n
    for n in outShape:
        numOutputs *= n

    boxSpecList = readVnnlib(vnnlibFilename, numInputs, numOutputs)
    targetAndType = findObjectiveFuncionType(boxSpecList[0][1], numOutputs)

    all_advinps = []
    all_non_advinps = [] # New list for non-adversarial inputs
    returnStatus = "timeout"

    for i in range(len(boxSpecList)):
        boxSpec = boxSpecList[i]
        inRanges = boxSpec[0]
        specList = boxSpec[1]
        random.seed()
        # Capture the third return value: non_advinps
        returnStatus, advinps, non_advinps = runSample(onnxModel, numInputs, numOutputs, inRanges, targetAndType, specList, inpDtype, inpShape)
        all_advinps.extend(advinps)
        
        # Collect the non-adversarial inputs
        all_non_advinps.extend(non_advinps)

        if returnStatus == "violated":
            # Continue collecting, do not return early
            pass
            
    # Return all three results
    return returnStatus, all_advinps, all_non_advinps