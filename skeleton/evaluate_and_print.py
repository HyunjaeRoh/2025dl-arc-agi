import numpy as np
from tqdm.auto import tqdm
import os
from evaluate import load_data, check_match
from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import json
from utils import render_grid
from rich import print as rich_print


def main():
    token = os.environ.get("HF_TOKEN", None)
    from arc import ARCSolver

    solver = ARCSolver(token=token, training=False)
    solver.prepare_evaluation()

    set_seed(1234567890)

    data_path = "/workspace/dataset"
    N_data = 10

    scores = []
    df = load_data(data_path)

    from datasets import Dataset
    #eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(N_data))
    eval_dataset = Dataset.from_pandas(df).shuffle(0).select(range(N_data))

    for i, eval_data in enumerate(tqdm(eval_dataset, desc="Evaluating Tasks")):  # tqdm에 desc 추가 및 인덱스 사용
        task_name = eval_data.get('task', f'Task_{i + 1}')  # 태스크 이름 가져오기 (없으면 기본값)
        rich_print(f"\n[bold cyan]----- Evaluating Task: {task_name} ----- [/bold cyan]")

        train_examples = eval_data["train"]
        test_input_grid = eval_data["test"][0]["input"]
        ground_truth_output_grid = eval_data["test"][0]["output"]

        if train_examples:
            rich_print(f"[bold magenta]Train Examples ({len(train_examples)} pairs):[/bold magenta]")
            for idx, train_ex in enumerate(train_examples):
                rich_print(f"[bold magenta]Train Example {idx} (Input):[/bold magenta]")
                render_grid(train_ex['input'])
                rich_print(f"[bold magenta]Train Example {idx} (Output):[/bold magenta]")
                render_grid(train_ex['output'])
                if idx < len(train_examples) - 1:  # 마지막 예제가 아니면 구분선 추가
                    rich_print("-" * 30)
        render_grid(test_input_grid)

        # 모델 예측
        predicted_output_grid = solver.predict(
            train_examples,
            test_input_grid,
        )

        rich_print("[bold green]Predicted Output:[/bold green]")
        render_grid(predicted_output_grid)

        rich_print("[bold red]Ground Truth Output:[/bold red]")
        render_grid(ground_truth_output_grid)

        s = check_match(predicted_output_grid, ground_truth_output_grid)
        scores.append(s)
        rich_print(f"[bold yellow]Match Score for this task: {s}[/bold yellow]")
        rich_print(f"[bold cyan]----- End of Task: {task_name} ----- [/bold cyan]")

    score = np.array(scores).mean() * 100
    print(f"\nEvaluation scores: {score:.2f}", flush=True)  # 최종 점수 전 줄바꿈 추가
    print("Evaluation Success")

if __name__ == "__main__":
    main()
