import csv

csv_file_path = 'whole_new_validated_ds_new_group_ids.csv'  # Replace with the path to your CSV file
output_file_path = 'whole_new_validated_ds_new_group_ids.txt'  # Replace with the desired output file path

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Create a list to store formatted lines
    formatted_lines = []

    # Iterate through each row in the CSV
    for row in csv_reader:
        formatted_row = ','.join(f'"{value}"' for value in row)
        formatted_lines.append(f'({formatted_row}),\n')

# Open a text file for writing and write the formatted lines
with open(output_file_path, 'w') as txt_file:
    txt_file.writelines(formatted_lines)