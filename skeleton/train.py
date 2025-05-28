import argparse
import os
import sys

from arc.arc import ARCSolver
from traceback import print_exc

def main():
    parser = argparse.ArgumentParser(description="Train ARCSolver model.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="../dataset",
        help="Path to the directory containing training JSON files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/arc_solver_finetuned_from_script",
        help="Directory to save the fine-tuned model adapter and training artifacts."
    )

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", None)

    print(f"Initializing ARCSolver with token: {'Provided' if token else 'Not provided'}")
    try:
        solver = ARCSolver(token=token, is_training=True)
    except Exception as e:
        print(f"Error initializing ARCSolver: {e}")
        sys.exit(1)

    print(f"Starting training with data from: {args.train_data_path}")
    print(f"Output will be saved to: {args.output_dir}")

    try:
        solver.train(
            training_data_path=args.train_data_path,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print_exc()
        sys.exit(1)

    print("Training script finished.")
   ###
if __name__ == "__main__":
    main()