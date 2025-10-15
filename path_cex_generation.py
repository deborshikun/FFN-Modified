import os
import subprocess
from pathlib import Path

# Configuration
GENERATED_PROPS_DIR = Path("generated_properties")
ONNX_PATH = "benchmarks/acasxu/ACASXU_run2a_2_9_batch_2000.onnx"
TIMEOUT = 300
NUM_LOOPS = 30

def run_verification_for_property(prop_file, output_folder):
    """Run verification for a single property file and save results to output folder"""
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    prop_path = str(prop_file)
    prop_name = prop_file.stem  # e.g., "prop_8_path_1_UNSAT"
    
    print(f"\n{'='*80}")
    print(f"Processing: {prop_name}")
    print(f"{'='*80}\n")
    
    # Run verification loops
    for i in range(NUM_LOOPS):
        print(f"Running Loop {i} for {prop_name}...")
        
        output_file = output_folder / f"{prop_name}__Loop{i}.txt"
        
        # Run the single instance script
        cmd = [
            "python",
            "run_single_instance.py",
            ONNX_PATH,
            prop_path,
            str(TIMEOUT)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT + 10  # Give a bit more time than the internal timeout
            )
            
            # Write output to file
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
            
            print(f"  Loop {i} completed: {output_file}")
            
        except subprocess.TimeoutExpired:
            print(f"  Loop {i} timed out")
            with open(output_file, 'w') as f:
                f.write("Status: timeout\n")
        except Exception as e:
            print(f"  Loop {i} error: {e}")
            with open(output_file, 'w') as f:
                f.write(f"Status: error\nError: {e}\n")
    
    print(f"\nCompleted all loops for {prop_name}")
    return output_folder

def merge_results(output_folder, prop_name):
    """Merge adversarial and non-adversarial results"""
    
    print(f"\nMerging results for {prop_name}...")
    
    adv_inputs = []
    nonadv_inputs = []
    
    # Collect all adversarial and non-adversarial inputs
    for i in range(NUM_LOOPS):
        loop_file = output_folder / f"{prop_name}__Loop{i}.txt"
        
        if not loop_file.exists():
            continue
        
        with open(loop_file, 'r') as f:
            lines = f.readlines()
        
        in_adversarial = False
        for line in lines:
            line_stripped = line.strip()
            
            if "Adversarial inputs found:" in line:
                in_adversarial = True
                continue
            elif "Status:" in line or line_stripped == "":
                in_adversarial = False
                continue
            
            if in_adversarial and line_stripped.startswith('[') and line_stripped.endswith(']'):
                adv_inputs.append(line_stripped)
    
    # Remove duplicates
    adv_inputs = list(set(adv_inputs))
    nonadv_inputs = list(set(nonadv_inputs))
    
    # Write merged files
    adv_merged_file = output_folder / "adv_merged.txt"
    nonadv_merged_file = output_folder / "nonadv_merged.txt"
    
    with open(adv_merged_file, 'w') as f:
        for inp in adv_inputs:
            f.write(inp + '\n')
    
    with open(nonadv_merged_file, 'w') as f:
        for inp in nonadv_inputs:
            f.write(inp + '\n')
    
    print(f"  Adversarial inputs: {len(adv_inputs)}")
    print(f"  Non-adversarial inputs: {len(nonadv_inputs)}")
    print(f"  Saved to: {output_folder}")
    
    return len(adv_inputs), len(nonadv_inputs)

def main():
    """Main execution function"""
    
    print("="*80)
    print("Automated Verification for Generated Properties")
    print("="*80)
    
    # Get all .vnnlib files in generated_properties
    vnnlib_files = sorted(GENERATED_PROPS_DIR.glob("*.vnnlib"))
    
    if not vnnlib_files:
        print("No .vnnlib files found in generated_properties/")
        return
    
    print(f"\nFound {len(vnnlib_files)} property files to process\n")
    
    results_summary = []
    
    for prop_file in vnnlib_files:
        prop_name = prop_file.stem
        output_folder = GENERATED_PROPS_DIR / prop_name
        
        # Run verification
        run_verification_for_property(prop_file, output_folder)
        
        # Merge results
        adv_count, nonadv_count = merge_results(output_folder, prop_name)
        
        results_summary.append({
            'property': prop_name,
            'adversarial': adv_count,
            'non_adversarial': nonadv_count
        })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Property':<40} {'Adversarial':<15} {'Non-Adversarial':<15}")
    print("-"*80)
    
    for result in results_summary:
        print(f"{result['property']:<40} {result['adversarial']:<15} {result['non_adversarial']:<15}")
    
    print("="*80)
    print("\nAll properties processed!")

if __name__ == "__main__":
    main()