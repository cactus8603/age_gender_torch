import pandas as pd

def process_adience(input_file, output_file):
    df = pd.read_csv(input_file)

    def process_age(age):
        if ';' in age:
            start, end = map(int, age.split(';'))
            if end - start <= 9:
                return (start + end) / 2
            return None

    df['age'] = df['age'].apply(process_age)

    df = df.dropna(subset=['age'])

    df.to_csv(output_file, index=False)

def process_fairface(input_file, output_file):
    """
    處理CSV檔案中的年齡資料，篩選並轉換特定的年齡範圍，然後儲存為新的檔案。
    
    參數:
    - input_file: str，輸入的CSV檔案名稱
    - output_file: str，處理後的CSV檔案名稱
    """

    # 讀取CSV檔案
    df = pd.read_csv(input_file)

    # 處理age欄位
    def process_age(age):
        if 'more than 70' in age:
            return 75
        elif '-' in age:
            start, end = map(int, age.split('-'))
            if end - start <= 9:
                return (start + end) / 2
        return None  # 若不符合條件則標記為 None

    # 應用處理函式
    df['age'] = df['age'].apply(process_age)

    # 移除無效的資料
    df = df.dropna(subset=['age'])

    # 儲存成新CSV檔案
    df.to_csv(output_file, index=False)

    print(f"處理完成並儲存為 '{output_file}'")

# 使用範例
# process_fairface('your_file.csv', 'processed_file.csv')

if __name__ == '__main__':
    # process_fairface('./fairface/fairface_label_val.csv', './fairface/processed_fairface_val.csv')

    process_adience('./adience/annotations/adience_annotations.csv', './adience/annotations/processed_adience.csv')