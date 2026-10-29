"""Create the pure Q2-Q8 variants of each family from its downloaded source.

    python -m rohan.quantize_family --family qwen gemma llama
"""
import argparse

from rohan.run_family_sweep import BASE, FAMILIES, check_tools, family_configs
from src.quantize import quantize_model


def quantize_family(family: str) -> None:
    source, *variants = family_configs(family)
    if not source.model_file.is_file():
        raise SystemExit(
            f"Missing source: {source.model_file}. "
            f"Run: python -m rohan.run_family_sweep --family {family} --prepare"
        )
    for config in variants:
        if config.model_file.is_file():
            print(f"Exists, skipping: {config.model_file}")
            continue
        # Always quantize from the original source, never another variant.
        quantize_model(BASE.quantize_tool, source.model_file, config.model_file,
                       config.quantization_type, pure=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", nargs="+", choices=FAMILIES, required=True)
    args = parser.parse_args()
    check_tools()
    for family in dict.fromkeys(args.family):
        quantize_family(family)


if __name__ == "__main__":
    main()
