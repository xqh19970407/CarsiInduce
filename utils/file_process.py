import csv
import json

def save_csv(path:str, head:list, data_list:list):
    # 创建 CSV 文件并写入数据
    
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头
        writer.writerow(head)
        
        # 写入数据
        writer.writerows(data_list)

    print(f"CSV文件已保存,{len(data_list)}条， 位置在", path)
    
def read_csv(file_path):
    data_list = []
    with open(file_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader:
            data_list.append(row)
    return data_list


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def save_json_file(data, file_path):
   
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"JSON file saved successfully: {file_path}")
    


def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.rstrip('\n') for line in lines]
        return lines
    