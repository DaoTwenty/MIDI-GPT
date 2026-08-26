import torch
import json
import pathlib
from midigpt.inference.model import get_model_class
import midigpt._core as _core
from midigpt.attributes.base import AttributeAnalyzer

def rebundle(path):
    print(f"Rebundling {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Force arch to gpt2 if None
    if ckpt.get("arch") is None:
        ckpt["arch"] = "gpt2"
        
    model_cls = get_model_class(ckpt["arch"])
    
    # Load encoder config from the dict
    ec_json = json.dumps(ckpt["encoder_config"])
    ec = _core.EncoderConfig.from_json(ec_json)
    
    # Explicitly add the attribute domains that we know are in these models
    # yellow.pt attributes:
    # note_density_quantile (10)
    # polyphony_quantile min (10)
    # polyphony_quantile max (10)
    # note_duration_quantile min (6)
    # note_duration_quantile max (6)
    
    # We use AttributeAnalyzer to get the specs, but we have to bypass the bug
    # where multiple instances of same registry key overwrite each other.
    
    from midigpt.attributes import ATTRIBUTE_REGISTRY
    
    attrs = [
        ATTRIBUTE_REGISTRY["note_density_quantile"](),
        ATTRIBUTE_REGISTRY["polyphony_quantile"](mode="min"),
        ATTRIBUTE_REGISTRY["polyphony_quantile"](mode="max"),
        ATTRIBUTE_REGISTRY["note_duration_quantile"](mode="min"),
        ATTRIBUTE_REGISTRY["note_duration_quantile"](mode="max"),
    ]
    
    analyzer = AttributeAnalyzer(attrs)
    specs = analyzer.token_domain_specs()
    print(f"Adding specs: {specs}")
    
    # Clear existing domains and re-derive (to ensure PieceStart is correct etc)
    ec.derive_token_domains()
    ec.add_attribute_token_domains(specs)
    
    # Verify vocab size
    vocab = _core.Vocabulary(ec)
    print(f"New vocab size: {vocab.size()}")
    if vocab.size() != 647:
        raise ValueError(f"Expected 647 tokens, got {vocab.size()}")
        
    # Update the ckpt dict
    ckpt["encoder_config"] = json.loads(ec.to_json())
    
    # Save back
    torch.save(ckpt, path)
    print(f"Successfully rebundled {path}")

if __name__ == "__main__":
    rebundle("models/yellow.pt")
    rebundle("models/yellow_250k_fixed_v2.pt")
