import json
from pathlib import Path
json_path = r'C:\Users\mxm6551xx\docker_mount_dir\recording_2024-09-11_12-48-18\transforms.json'
# Open the JSON file for reading
with open(json_path, 'r') as f:
    data = json.load(f)


# Modify the data
for f in data['frames']:
    f['file_path'] = Path(f['file_path']).as_posix().replace('\\', '/')

# Write the modified data back to the file
with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)  # indent for pretty formatting