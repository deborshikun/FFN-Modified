input_file = "prop_8_data/adv_with_out_total.txt"
output_file = "prop_8_data/adv_merged_outputs_file.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.strip().split("=>")
        if len(parts) == 2:
            output_vector = parts[1].strip()
            outfile.write(output_vector + "\n")

print(f"Extracted outputs written to '{output_file}'")

input_file = "prop_8_data/adv_merged_outputs_file.txt"

count_WR = 0
count_SL = 0
count_SR = 0

minimum_WR = []
minimum_SL = []
minimum_SR = []

with open(input_file, "r") as infile:
    for line in infile:
        values = eval(line.strip())
        wr, sl, sr = values[2], values[3], values[4]
        min_val = min(wr, sl, sr)
        if min_val == wr:
            count_WR += 1
            minimum_WR.append(wr)
        elif min_val == sl:
            count_SL += 1
            minimum_SL.append(sl)
        else:  # min_val == sr
            count_SR += 1
            minimum_SR.append(sr)

print(f"WR minimum count: {count_WR}")
print(f"SL minimum count: {count_SL}")
print(f"SR minimum count: {count_SR}")

# Save the lists to output_splits.txt
with open("prop_8_data/output_splits.txt", "w") as f:
    f.write(str(minimum_WR) + "\n")
    f.write(str(minimum_SL) + "\n")
    f.write(str(minimum_SR) + "\n")

print("Minimum WR, SL, SR lists saved to 'output_splits.txt'")