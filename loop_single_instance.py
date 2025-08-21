import os

single = "run_single_instance.py"
# onnx_path = "benchmarks/acasxu/ACASXU_run2a_2_9_batch_2000.onnx"
# prop_path = "benchmarks/acasxu/prop_1.vnnlib"

onnx_name = input("Model Name: ")
prop_name = input("Property Name: ")

onnx_path = "benchmarks/acasxu/" + onnx_name + "_batch_2000.onnx"
prop_path = "benchmarks/acasxu/" + prop_name + ".vnnlib"

timeoutval = input("Enter timeout value in seconds (default is 300): ")
if not timeoutval:
    timeoutval = 300

i = 0
while i < 30:
    os.system(f"python {single} -m {onnx_path} -p {prop_path} -o {onnx_name}__{prop_name}__Loop{i}.txt -t {timeoutval}")
    i += 1

# Merging time
def merge_adversarial_files(onnx_name, prop_name, num_loops=30):
    """Merge adversarial inputs from all loop files"""
    merged_adv_content = []
    merged_nonadv_content = []
    
    for i in range(num_loops):
        # Check main adversarial file
        filename_adv = f"{onnx_name}__{prop_name}__Loop{i}.txt"
        # Check separate non-adversarial file
        filename_nonadv = f"{onnx_name}__{prop_name}__Loop{i}_non_adversarial.txt"
        
        # Process adversarial file
        try:
            print(f"Checking adversarial file: {filename_adv}")
            with open(filename_adv, 'r') as file:
                lines = file.readlines()
            
            print(f"File {filename_adv} has {len(lines)} lines")
            
            # Skip first 2 lines and process the rest
            if len(lines) > 2:
                content = ''.join(lines)  # Check full content for adversarial inputs
                
                # Check if this file contains adversarial inputs
                if "Adversarial inputs found:" in content:
                    print(f"Processing {filename_adv} - Found adversarial inputs section")
                    
                    # Look for lines that contain adversarial input arrays
                    in_adversarial_section = False
                    for line in lines:  # Process all lines, not just lines[2:]
                        line_stripped = line.strip()
                        
                        if "Adversarial inputs found:" in line:
                            in_adversarial_section = True
                            continue
                        elif line_stripped == "" or "Status:" in line:
                            in_adversarial_section = False
                            continue
                            
                        if in_adversarial_section and line_stripped.startswith('[') and line_stripped.endswith(']'):
                            merged_adv_content.append(line_stripped + '\n')
                            print(f"  Added adversarial input: {line_stripped[:50]}...")
                else:
                    print(f"No adversarial inputs found in {filename_adv}")
            
        except FileNotFoundError:
            print(f"Warning: {filename_adv} not found, skipping...")
        except Exception as e:
            print(f"Error processing {filename_adv}: {e}")
        
        # Process non-adversarial file
        try:
            print(f"Checking non-adversarial file: {filename_nonadv}")
            with open(filename_nonadv, 'r') as file:
                lines = file.readlines()
            
            print(f"File {filename_nonadv} has {len(lines)} lines")
            
            # Skip first line (status line) and process the rest
            if len(lines) > 1:
                print(f"Processing {filename_nonadv} - Found non-adversarial inputs")
                
                for line in lines[1:]:  # Skip first line which contains "Status: non-adversarial inputs found"
                    line_stripped = line.strip()
                    
                    if line_stripped.startswith('[') and line_stripped.endswith(']'):
                        merged_nonadv_content.append(line_stripped + '\n')
                        print(f"  Added non-adversarial input: {line_stripped[:50]}...")
            else:
                print(f"No non-adversarial inputs found in {filename_nonadv}")
            
        except FileNotFoundError:
            print(f"Warning: {filename_nonadv} not found, skipping...")
        except Exception as e:
            print(f"Error processing {filename_nonadv}: {e}")
    
    # Write merged adversarial file
    if merged_adv_content:
        with open('adv_merged.txt', 'w') as file:
            file.writelines(merged_adv_content)
        print(f"Created 'adv_merged.txt' with {len(merged_adv_content)} adversarial inputs")
    else:
        print("No adversarial inputs found to merge")
    
    # Write merged non-adversarial file
    if merged_nonadv_content:
        with open('nonadv_merged.txt', 'w') as file:
            file.writelines(merged_nonadv_content)
        print(f"Created 'nonadv_merged.txt' with {len(merged_nonadv_content)} non-adversarial inputs")
    else:
        print("No non-adversarial inputs found to merge")
    
    return len(merged_adv_content), len(merged_nonadv_content)
    
    return len(merged_adv_content), len(merged_nonadv_content)

# Perform the merging
adv_count, nonadv_count = merge_adversarial_files(onnx_name, prop_name)

print(f"\nMerging completed")
print(f"Total adversarial inputs merged: {adv_count}")
print(f"Total non-adversarial inputs merged: {nonadv_count}")
print(f"Files created: 'adv merged.txt' and 'nonadv merged.txt'")

# Remove duplicates from merged files
print("\nRemoving duplicates from merged files...")

def remove_duplicates_from_file(filename):
    """Remove duplicate lines from a file"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        unique_lines = []
        seen = set()
        for line in lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)
        
        # Write back the unique lines
        with open(filename, 'w') as file:
            file.writelines(unique_lines)
        
        removed_count = len(lines) - len(unique_lines)
        print(f"{filename}: Removed {removed_count} duplicates, {len(unique_lines)} unique entries remaining")
        
    except FileNotFoundError:
        print(f"File {filename} not found for duplicate removal")

if os.path.exists('adv_merged.txt'):
    remove_duplicates_from_file('adv_merged.txt')

if os.path.exists('nonadv_merged.txt'):
    remove_duplicates_from_file('nonadv_merged.txt')


