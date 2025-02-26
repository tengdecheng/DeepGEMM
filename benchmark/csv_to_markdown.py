import csv

def csv_to_markdown(input_csv_file, output_markdown_file):
    # 打开CSV文件并读取内容
    with open(input_csv_file, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # 读取表头
        headers = next(csvreader)
        
        # 构建Markdown表头
        markdown_table = "| " + " | ".join(headers) + " |\n"
        markdown_table += "| " + " | ".join([":---:"] * len(headers)) + " |\n"
        
        # 遍历CSV文件的每一行并添加到Markdown表格
        for row in csvreader:
            markdown_table += "| " + " | ".join(row) + " |\n"
    
    # 将Markdown表格写入文件
    with open(output_markdown_file, mode='w', encoding='utf-8') as mdfile:
        mdfile.write(markdown_table)

csv_to_markdown("perf.csv", "perf.markdown")