"""
Diagnostic script to analyze token distribution and class imbalance.
"""
import torch
import albumentations as A
from datasets.mp100_cape import MP100CAPE
from datasets.token_types import TokenType

def main():
    num_samples = 20
    
    # Build dataset directly
    print("Loading dataset...")
    transforms = A.Compose([
        A.Resize(512, 512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_train.json',
        transforms=transforms,
        split='train',
        vocab_size=2000,
        seq_len=200
    )
    
    # Analyze token distribution
    total_coord = 0
    total_sep = 0
    total_eos = 0
    total_visible_coord = 0
    total_visible_sep = 0
    total_visible_eos = 0
    
    print(f"\nAnalyzing {num_samples} samples...")
    print("=" * 80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        seq_data = sample['seq_data']
        
        token_labels = seq_data['token_labels']
        visibility_mask = seq_data['visibility_mask']
        
        # Count tokens
        coord_count = (token_labels == TokenType.coord.value).sum().item()
        sep_count = (token_labels == TokenType.sep.value).sum().item()
        eos_count = (token_labels == TokenType.eos.value).sum().item()
        
        # Count visible tokens (those included in loss)
        coord_mask = (token_labels == TokenType.coord.value) & visibility_mask
        sep_mask = (token_labels == TokenType.sep.value) & visibility_mask
        eos_mask = (token_labels == TokenType.eos.value) & visibility_mask
        
        visible_coord = coord_mask.sum().item()
        visible_sep = sep_mask.sum().item()
        visible_eos = eos_mask.sum().item()
        
        total_coord += coord_count
        total_sep += sep_count
        total_eos += eos_count
        total_visible_coord += visible_coord
        total_visible_sep += visible_sep
        total_visible_eos += visible_eos
        
        if i < 3:  # Print first 3 samples
            print(f"\nSample {i}:")
            print(f"  Category: {sample.get('category_id', 'N/A')}")
            print(f"  Token counts (all):")
            print(f"    COORD: {coord_count}")
            print(f"    SEP:   {sep_count}")
            print(f"    EOS:   {eos_count}")
            print(f"  Token counts (visible/in loss):")
            print(f"    COORD: {visible_coord}")
            print(f"    SEP:   {visible_sep}")
            print(f"    EOS:   {visible_eos}")
            print(f"  ⚠️  Class imbalance ratio (COORD:EOS): {visible_coord}:{visible_eos}")
    
    print("\n" + "=" * 80)
    print(f"OVERALL STATISTICS (across {num_samples} samples):")
    print("=" * 80)
    print(f"\nTotal tokens:")
    print(f"  COORD: {total_coord}")
    print(f"  SEP:   {total_sep}")
    print(f"  EOS:   {total_eos}")
    print(f"\nTotal visible tokens (included in loss):")
    print(f"  COORD: {total_visible_coord}")
    print(f"  SEP:   {total_visible_sep}")
    print(f"  EOS:   {total_visible_eos}")
    
    if total_visible_eos > 0:
        coord_eos_ratio = total_visible_coord / total_visible_eos
        sep_eos_ratio = total_visible_sep / total_visible_eos
        print(f"\n🔴 CRITICAL CLASS IMBALANCE:")
        print(f"  COORD tokens appear {coord_eos_ratio:.1f}× more often than EOS")
        print(f"  SEP tokens appear {sep_eos_ratio:.1f}× more often than EOS")
        print(f"\n💡 Impact:")
        print(f"  - Model receives {coord_eos_ratio:.1f}× more gradient signal for COORD")
        print(f"  - Model receives only 1× gradient signal for EOS")
        print(f"  - Result: Model learns to predict COORD easily, but struggles with EOS")
        print(f"\n✅ Solution:")
        print(f"  - Use class-weighted cross-entropy loss")
        print(f"  - Give EOS token weight = {coord_eos_ratio:.1f}")
        print(f"  - Give COORD/SEP token weight = 1.0")

if __name__ == '__main__':
    main()

