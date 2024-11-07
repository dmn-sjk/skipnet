import yaml
import tabulate
import argparse
import os
 

results_file = 'metric.yaml'
data_task_id = 9


def process_results(folders):
    data = []
    for folder in folders:
        results = load_results(folder)
        results = results[data_task_id]
        # results = results[0]

        row = {}
        row['Name'] = folder.split('/')[-1]
        # for i in range(data_task_id + 1):
        for i in range(data_task_id):
            row[f'Acc_task_{i}'] = f"{{:.{1}f}} ({{:.{1}f}})".format(
                results[f'task_{i}_acc'],
                results[f'task_{i}_acc_diff_init']
                )
            # row[f'Acc_task_{i}'] = f"{{:.{1}f}}".format(
            #     results[f'task_{i}_acc']
            #     )
        row[f'Acc_task_{data_task_id}'] = f"{{:.1f}} (0.0)".format(results['curr_acc']) 
        data.append(row)
    
    # Generate a table with headers
    table = tabulate.tabulate(data, headers={key: key for key in row.keys()}, tablefmt="grid")
    print(table)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_folders', nargs='+', type=str)
    return parser.parse_args()

def load_results(folder):
    results_path = os.path.join(folder, results_file)
    with open(results_path, 'r') as f:
        return yaml.load(f)

if __name__ == "__main__":
    args = parse_args()
    process_results(args.result_folders)