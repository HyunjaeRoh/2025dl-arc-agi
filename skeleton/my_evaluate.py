import numpy as np
from tqdm.auto import tqdm
import os
from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import json
from utils import render_grid
from rich import print as rich_print
from datasets import Dataset
from evaluate import load_data as __load_data


def load_data_selective(base_dir, task_list, window, rng, max_len=1000):
    all_filenames = os.listdir(base_dir)

    json_filenames = [f for f in all_filenames if f.endswith(".json")]

    json_filenames.sort()

    selected_filenames = []
    for i, filename in enumerate(json_filenames):
        if i in task_list:
            selected_filenames.append(filename)

    if not selected_filenames:
        print("경고: task_list에 해당하는 작업이 선택되지 않았습니다.")
        return pd.DataFrame()

    print(f"선택된 작업 파일 ({len(selected_filenames)}개): {selected_filenames}")

    # 선택된 작업 데이터 로드
    dataset = []
    filenames_map = {}
    for fn in selected_filenames:
        filepath = os.path.join(base_dir, fn)
        try:
            with open(filepath) as fp:
                task_data = json.load(fp)
                # 작업 데이터가 리스트 형태인지 확인 (예제들의 리스트)
                if isinstance(task_data, list):
                    dataset.append(task_data)
                    filenames_map[len(dataset) - 1] = fn.split(".")[0]
                else:
                    print(f"경고: {fn} 파일의 형식이 예상과 다릅니다. (리스트가 아님) - 건너뜁니다.")
        except Exception as e:
            print(f"오류: {fn} 파일 로드 중 오류 발생: {e} - 건너뜁니다.")

    if not dataset:
        print("오류: 유효한 작업 데이터를 로드하지 못했습니다.")
        return pd.DataFrame()

    data = []
    N_selected_tasks = len(dataset)
    train_size = window - 1
    test_size = 1

    if train_size < 0:
        print("오류: window 크기는 최소 1이어야 합니다.")
        return pd.DataFrame()

    # max_len 만큼 데이터 포인트 생성
    while len(data) < max_len:
        # 선택된 작업 목록에서 무작위로 하나 선택
        task_idx = rng.integers(0, N_selected_tasks)
        task = dataset[task_idx]
        file_name = filenames_map[task_idx]
        n_examples = len(task)

        # 4. window 크기로 예제 샘플링
        if n_examples < window:
            print(f"경고: 작업 {file_name}의 예제 수({n_examples})가 window({window})보다 작습니다. 복원 추출을 사용합니다.")
            replace = True
        else:
            replace = False

        # window 크기만큼 인덱스 샘플링
        grids_idx = rng.choice(n_examples, size=window, replace=replace)
        train_grids = [task[i] for i in grids_idx[:train_size]]
        test_grids = [task[i] for i in grids_idx[train_size:]] # 항상 1개

        # 데이터 형식 구성
        test_inputs = [{'input': grid['input']} for grid in test_grids]
        test_outputs = [grid['output'] for grid in test_grids]
        test_outputs_transformed = [{'output': grid} for grid in test_outputs]
        combined_tests = []
        for test_input, test_output in zip(test_inputs, test_outputs_transformed):
            combined_tests.append({'input': test_input['input'], 'output': test_output['output']})

        data.append({
            'task': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs,
            'test': combined_tests,
        })

    df = pd.DataFrame(data)
    return df



def main():
    token = os.environ.get("HF_TOKEN", None)
    from arc.arc import ARCSolver

    solver = ARCSolver(token=token, is_training=False)
    solver.prepare_evaluation(adapter_path="artifacts/arc_solver_finetuned_from_script/checkpoint-epoch-1")
    #solver.prepare_evaluation()

    set_seed(1234567890)
    rng = np.random.default_rng(42)

    data_path = "../dataset"
    N_data = 10

    task_list = [1]

    #df = load_data_selective(data_path, [0], window=3, rng=rng, max_len=10)
    df = __load_data(data_path)



    eval_dataset = Dataset.from_pandas(df).select(range(N_data))
    eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(N_data))

    #eval_dataset =

    for i, eval_data in enumerate(tqdm(eval_dataset, desc="Evaluating Tasks")):  # tqdm에 desc 추가 및 인덱스 사용
        task_name = eval_data.get('task', f'Task_{i + 1}')  # 태스크 이름 가져오기 (없으면 기본값)
        rich_print(f"\n[bold cyan]----- Evaluating Task: {task_name} ----- [/bold cyan]")

        train_examples = eval_data["train"]
        test_input_grid = eval_data["test"][0]["input"]
        ground_truth_output_grid = eval_data["test"][0]["output"]

        if train_examples:
            rich_print(f"[bold magenta]Train Examples ({len(train_examples)} pairs):[/bold magenta]")
            for idx, train_ex in enumerate(train_examples[:3]):
                rich_print(f"[bold magenta]Train Example {idx} (Input):[/bold magenta]")
                render_grid(train_ex['input'])
                rich_print(f"[bold magenta]Train Example {idx} (Output):[/bold magenta]")
                render_grid(train_ex['output'])
                if idx == len(train_examples) - 1:  # 마지막 예제가 아니면 구분선 추가
                    rich_print("-" * 30)
        render_grid(test_input_grid)

        # 모델 예측
        predicted_output_grid, rule, flags, total_outputs = solver.predict(
            train_examples,
            test_input_grid,
            get_details=True,
        ) ##

        rich_print(f"@@ flags: {flags}")
        print(f"@@ predicted_rule: {rule} @@@")
        print(f"@@ total_outputs: {total_outputs}")

        rich_print("[bold green]Predicted Output:[/bold green]")
        render_grid(predicted_output_grid)

        rich_print("[bold red]Ground Truth Output:[/bold red]")
        render_grid(ground_truth_output_grid)



if __name__ == "__main__":
    main()
