import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    # Step 1: Read classes.txt
    if not os.path.exists(path_to_classes_txt):
        print(f'❌ classes.txt not found at {path_to_classes_txt}')
        return
    
    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # Step 2: Create the YAML config
    data = {
        'path': './custom_data',  # relative to where training is run
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }

    # Step 3: Write it to file
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f'✅ Created Ultralytics config at {path_to_data_yaml}\n')
    print('Contents:')
    print('----------')
    print(yaml.dump(data, sort_keys=False))

# Update these paths if needed
path_to_classes_txt = '/workspace/data/YOLO/custom_data/classes.txt'
path_to_data_yaml = '/workspace/data/YOLO/data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)
