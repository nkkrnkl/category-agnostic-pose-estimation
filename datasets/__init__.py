import torch.utils.data
from .mp100_cape import build_mp100_cape
def build_dataset(image_set, args):
    if args.semantic_classes > 0:
        pass
    if args.dataset_name == 'mp100':
        print(f"Build MP-100 CAPE {image_set} dataset")
        return build_mp100_cape(image_set, args)
    elif args.dataset_name in ['stru3d', 'scenecad', 'rplan', 'cubicasa', 'waffle', 'r2g']:
        raise ValueError(f'Polygon datasets not supported in CAPE. Only MP-100 is supported. '
                        f'(poly_data.py was removed as part of CAPE implementation cleanup)')
    raise ValueError(f'dataset {args.dataset_name} not supported')
def get_dataset_class_labels(dataset_name):
    semantics_label = None
    if dataset_name == 'stru3d':
        semantics_label = {
            0: 'Living Room',
            1: 'Kitchen',
            2: 'Bedroom',
            3: 'Bathroom',
            4: 'Balcony',
            5: 'Corridor',
            6: 'Dining room',
            7: 'Study',
            8: 'Studio',
            9: 'Store room',
            10: 'Garden',
            11: 'Laundry room',
            12: 'Office',
            13: 'Basement',
            14: 'Garage',
            15: 'Misc.',
            16: 'Door',
            17: 'Window'
        }
    elif dataset_name == 'cubicasa':
        semantics_label = {
            "Outdoor": 0,
            "Kitchen": 1,
            "Living Room": 2,
            "Bed Room": 3,
            "Bath": 4,
            "Entry": 5,
            "Storage": 6,
            "Garage": 7,
            "Undefined": 8,
            "Window": 9,
            "Door": 10,
        }
    elif dataset_name == 'r2g':
        semantics_label = {
            'unknown': 0, 
            'living_room': 1, 
            'kitchen': 2, 
            'bedroom': 3, 
            'bathroom': 4, 
            'restroom': 5, 
            'balcony': 6, 
            'closet': 7, 
            'corridor': 8, 
            'washing_room': 9, 
            'PS': 10, 
            'outside': 11}
    id2class = {v: k for k, v in semantics_label.items()} if semantics_label else None
    return semantics_label, id2class