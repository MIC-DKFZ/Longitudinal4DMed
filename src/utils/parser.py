import argparse

def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to use --config. Run: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal Flow Matching (discrete) training")

    # data
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)

    # model
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--num-levels", type=int, default=4,
                        help="Number of down/up levels in the UNet (depends on your implementation).")
    parser.add_argument('--number_evals', type=int, default=10)
    parser.add_argument('--training_noise', type=float, default=0.01)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)

    # misc
    parser.add_argument('--debug', action='store_true', help='If set, run in debug mode.')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset", type=str, default="isles",
                        help="Dataset to use. Only used if --dummy is not set.") # acdc, isles, lumiere, oasis
    parser.add_argument("--model_type", type=str, default="tfm")
    parser.add_argument("--fm_model_unet_expands", nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument("--log_dir", type=str, default=None,
                        help="TensorBoard log directory. Defaults to <save_dir>/logs.")
    parser.add_argument("--dummy", action="store_true",
                        help="Use DummyTemporalDataset instead of a real dataset.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file. CLI flags override YAML values.")

    # Pre-parse to find --config, then apply YAML as defaults before full parse.
    pre, _ = parser.parse_known_args()
    if pre.config is not None:
        yaml_cfg = _load_yaml_config(pre.config)
        parser.set_defaults(**yaml_cfg)

    return parser.parse_args()