import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal Flow Matching (discrete) training")

    # data
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)

    # model
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-levels", type=int, default=4,
                        help="Number of down/up levels in the UNet (depends on your implementation).")
    parser.add_argument('--number_evals', type=int, default=10)

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
                        help="Dataset to use. Only used if --dummy is not set.") # acdc, isles, lumiere
    parser.add_argument("--dummy", action="store_true",
                        help="Use DummyTemporalDataset instead of a real dataset.")

    return parser.parse_args()