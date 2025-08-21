import sys
import time
import signal_windows as signal #modified for windows
import argparse

# --- Ensure you have this import for the path to work correctly ---
from src.FFNEvaluation import sampleEval, supersats, superunsats

# Global variable for timeout
system_timeout = 60.0  # Default value

# Register an handler for the timeout
def handler(signum, frame):
    raise Exception("")#kill running :: Timeout occurs")

# --- MODIFIED FUNCTION ---
def runSingleInstanceForAllCategory(onnxFile, vnnlibFile, timeout_val):
   'called from run_all_catergory.py'
   
   # Use a local timeout value
   local_timeout = float(timeout_val)
   
   try:
       # Ignore the second return value (non_adv_inputs) to maintain compatibility
       retStatus, _ = runSingleInstance(onnxFile, vnnlibFile, local_timeout)
       return retStatus
   except Exception as exc:
       print(exc)
       return "timeout," + str(local_timeout) + "\n"

# --- MODIFIED FUNCTION ---
def runSingleInstance(onnxFile, vnnlibFile, timeout_duration=None):
   # If no timeout provided, use the system default
   if timeout_duration is None:
       timeout_duration = system_timeout
   
   # Variable Initialization
   startTime = time.time()
   all_adv_inputs = []
   all_non_adv_inputs = [] # New list for non-adversarial inputs
   
   onnxFileName = onnxFile.split('/')[-1]
   vnnFileName = vnnlibFile.split('/')[-1]
   
   # Calculate end time based on provided timeout
   end_time = startTime + float(timeout_duration)
   
   iteration = 0
   while time.time() < end_time:
       iteration += 1
       
       # Call sampleEval to find inputs
       # It now returns three values; we capture all of them
       status, adv_inputs, non_adv_inputs = sampleEval(onnxFile, vnnlibFile)
       
       # Add any found adversarial inputs to our collection
       if adv_inputs and len(adv_inputs) > 0:
           for inp in adv_inputs:
               if inp not in all_adv_inputs:
                   all_adv_inputs.append(inp)
       
       # Add any found non-adversarial inputs to our collection
       if non_adv_inputs and len(non_adv_inputs) > 0:
            for inp in non_adv_inputs:
                if inp not in all_non_adv_inputs:
                    all_non_adv_inputs.append(inp)
   
   # Calculate total time used
   timeElapsed = time.time() - startTime
   
   # Prepare result string
   result = "" 
   
   if all_adv_inputs:
       result += f"Status: violated\n"
       result += "Adversarial inputs found:\n"
       for adv_input in all_adv_inputs:
           result += f"  {adv_input}\n"
   else:
       result += f"Status: timeout\n"
   
   # Return both the result string and the list of non-adversarial inputs
   return result, all_non_adv_inputs


#Main function
if __name__ == '__main__':
   # Parse arguments
   parser = argparse.ArgumentParser()
   parser.add_argument('-m', help='A required onnx model file path')
   parser.add_argument('-p', help='A required vnnlib file path')
   parser.add_argument('-o', help='An optional result file path')
   parser.add_argument('-t', help='An optional timeout')

   args = parser.parse_args()
   onnxFile = args.m
   vnnlibFile = args.p
   
   if (onnxFile is None):
      print ("\n!!! Failed to provide onnx file on the command line!")
      sys.exit(1)

   if (vnnlibFile is None):
      print ("\n!!! Failed to provide vnnlib file path on the command line!")
      sys.exit(1)

   resultFile = args.o 

   if ( resultFile is None ):
      resultFile = "out.txt"
      print ("\n!!! No result_file path is provided on the command line!")
      print(f"Output will be written in default result file- \"{resultFile}\"")
   else:
      print(f"\nOutput will be written in - \"{resultFile}\"")

   # --- NEW: Define filename for non-adversarial inputs ---
   if '.' in resultFile:
       nonAdvResultFile = resultFile.rsplit('.', 1)[0] + "_non_adversarial.txt"
   else:
       nonAdvResultFile = resultFile + "_non_adversarial"
   print(f"Non-adversarial inputs will be written in - \"{nonAdvResultFile}\"")


   timeout_arg = args.t

   if (timeout_arg is None):
      print ("\n!!! timeout is not on the command line!")
      print ("Default timeout is set as - 60 sec")
      cmd_timeout = 60.0
   else:
      print (f"\ntimeout is  - {timeout_arg} sec")
      cmd_timeout = float(timeout_arg)

   # Register the signal function handler
   signal.signal(signal.SIGALRM, handler)
   signal.alarm(int(cmd_timeout))
   
   outFile = open(resultFile, "w")
   # --- NEW: Open the second file for writing ---
   nonAdvOutFile = open(nonAdvResultFile, "w")

   try:
       # runSingleInstance now returns two values
       retStatus, non_adversarial_inputs = runSingleInstance(onnxFile, vnnlibFile, cmd_timeout)
       
       # Write adversarial results to the original file
       outFile.write(retStatus)
       print(f"\nOutput is written in - \"{resultFile}\"")

       # --- NEW: Write non-adversarial inputs to the new file ---
       if non_adversarial_inputs:
           nonAdvOutFile.write("Status: non-adversarial inputs found\n")
           for inp in non_adversarial_inputs:
               nonAdvOutFile.write(str(inp) + "\n")
       else:
           nonAdvOutFile.write("Status: no non-adversarial inputs found\n")
       print(f"Non-adversarial output is written in - \"{nonAdvResultFile}\"")


   except Exception as exc:
       print(exc)
       outFile.write("timeout," + str(cmd_timeout) + "\n")
       nonAdvOutFile.write("timeout," + str(cmd_timeout) + "\n") # Also note timeout in the new file
       print(f"\nOutput is written in - \"{resultFile}\"")
       print(f"\n!!! Timeout occurred after {cmd_timeout} seconds")
   
   finally:
        # Ensure both files are closed
        outFile.close()
        nonAdvOutFile.close()