import torch
from torchvision import models
from collections import OrderedDict

def diagnose_checkpoint(checkpoint_path):
    print("=" * 60)
    print("CHECKPOINT DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✓ Found 'state_dict' in checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✓ Found 'model_state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            print("✓ Using checkpoint directly as state_dict")
        
        # Show other keys
        other_keys = [k for k in checkpoint.keys() if k not in ['state_dict', 'model_state_dict']]
        if other_keys:
            print(f"Other keys in checkpoint: {other_keys}")
    else:
        state_dict = checkpoint
        print("✓ Checkpoint is a state_dict")
    
    # Get model architecture from keys
    model_type = None
    for key in state_dict.keys():
        if 'denseblock1' in key:
            model_type = 'densenet'
            break
    
    # Count keys
    print(f"Number of keys in state_dict: {len(state_dict)}")
    
    # Show the first few keys
    print("\nFirst 10 keys in checkpoint:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key} -> {state_dict[key].shape}")
    
    # Check for common issues
    print("\n" + "=" * 60)
    print("POTENTIAL ISSUES:")
    print("=" * 60)
    
    # Check if classifier is present
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
    if classifier_keys:
        print(f"✓ Found classifier keys: {len(classifier_keys)}")
        for key in classifier_keys:
            print(f"  - {key}: {state_dict[key].shape}")
    else:
        print("⚠ No classifier keys found - likely a feature extractor only")
    
    # Check for pretrained keys (conv0, norm0)
    pretrained_keys = [k for k in state_dict.keys() if 'conv0' in k or 'norm0' in k]
    if pretrained_keys:
        print(f"✓ Found pretrained feature extractor keys: {len(pretrained_keys)}")
    else:
        print("⚠ No pretrained feature extractor keys found")
    
    # Check for batch norm stats
    bn_stats = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k]
    if bn_stats:
        print(f"✓ Found batch norm statistics: {len(bn_stats)}")
    else:
        print("⚠ No batch norm statistics found - model may be in eval mode")
    
    # Return the state_dict for further inspection
    return state_dict

# Run the diagnostic
state_dict = diagnose_checkpoint(r'C:\PyTorch\CNN_in_PyTorch\checkpoints\ckpt_2cl_pretr_densenet121_e23_bal0.860_comp0.856_ds3_NEW_BEST.pt')

# Now try to load with different settings
def test_load(state_dict):
    model_architectures = [
        ('densenet121', models.densenet121),
        ('densenet169', models.densenet169),
        ('densenet201', models.densenet201),
        ('densenet161', models.densenet161)
    ]
    
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT ARCHITECTURES:")
    print("=" * 60)
    
    for name, model_fn in model_architectures:
        try:
            model = model_fn(pretrained=False)
            # Try loading with strict=False first
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            print(f"\n{name}:")
            print(f"  ✓ Loaded successfully (strict=False)")
            if missing:
                print(f"  Missing keys: {len(missing)}")
                # Show first few missing keys
                for key in list(missing)[:3]:
                    print(f"    - {key}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
                for key in list(unexpected)[:3]:
                    print(f"    - {key}")
            
            # Try strict loading
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"  ✓ Also works with strict=True!")
            except Exception as e:
                print(f"  ✗ strict=True fails: {str(e)[:100]}...")
                
        except Exception as e:
            print(f"\n{name}: ✗ Failed to load: {str(e)[:100]}...")

test_load(state_dict)