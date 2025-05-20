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
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for restricted models like Llama 3. Can also be set via HUGGING_FACE_HUB_TOKEN environment variable."
    )
    # 추가적인 학습 파라미터 (예: epochs, learning_rate 등)를 여기서 받을 수 있습니다.
    # 이 경우 ARCSolver.train 메서드도 해당 파라미터를 받도록 수정해야 합니다.

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", None)

    print(f"Initializing ARCSolver with token: {'Provided' if token else 'Not provided'}")
    try:
        solver = ARCSolver(token=token)
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