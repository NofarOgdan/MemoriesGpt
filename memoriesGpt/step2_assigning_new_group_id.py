import csv
import uuid

csv_file_path = 'whole_new_validated_ds.csv'  # Replace with the path to your CSV file
output_file_path = 'whole_new_validated_ds_new_group_ids.csv'  # Replace with the desired output file path

uuid_mapping = {}  # To store mapping of old UUIDs to new UUIDs


def generate_new_uuid(old_uuid):
    if old_uuid not in uuid_mapping:
        uuid_mapping[old_uuid] = str(uuid.uuid4())
    return uuid_mapping[old_uuid]


with open(csv_file_path, 'r') as csv_file, open(output_file_path, 'w', newline='') as output_file:
    csv_reader = csv.DictReader(csv_file)
    fieldnames = csv_reader.fieldnames

    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in csv_reader:
        old_uuid = row['group_id']  # Replace 'UUID_column' with the actual column name
        new_uuid = generate_new_uuid(old_uuid)
        row['group_id'] = new_uuid
        csv_writer.writerow(row)
