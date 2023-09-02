import re
import opencc

"""
格式化DRCD-backtrans
"""

# 读取保存内容的文本文件
with open('data/raw/DRCD-backtrans.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()

# 使用正则表达式提取问题和内容
pattern = r'question: (.*?)\n.*?text: (.*?)\n'
matches = re.findall(pattern, input_text, re.DOTALL)

# 将匹配结果整理成纯文本格式
output_text = ""
for match in matches:
    output_text += f"{match[0]}\n{match[1]}\n"

# 将结果写入一个新的纯文本文件
with open('data/temporary/DRCD-backtrans/traditional.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(output_text)

path = 'data/temporary/DRCD-backtrans/traditional.txt'
dataFile = open(path, 'r', encoding='utf8')

qaList = dataFile.readlines()
total = []

# 逐行读取数据，并将每两行作为一个问答对
for i in range(0, len(qaList), 4):
    question = qaList[i].strip()
    answer = qaList[i + 1].strip()
    qa = [question, answer]
    total.append(qa)

# 关闭原始数据文件
dataFile.close()

resultData = open('data/temporary/DRCD-backtrans/traditional.txt', 'w', encoding='utf8')

# 遍历处理后的问答对列表，写入格式化后的数据文件
for i in total:
    question = i[0].replace('"', '')
    answer = i[1].replace('"', '')
    qa = question + '|' + answer
    resultData.write(qa + '\n')
    # 关闭结果数据文件
resultData.close()

print('DRCD-backtrans初步转换完成！')

def convert_traditional_to_simplified(input_path, output_path):
    cc = opencc.OpenCC("t2s")  # 创建一个转换器，从繁体转换为简体

    with open(input_path, "r", encoding="utf-8") as input_file:
        traditional_text = input_file.read()

    simplified_text = cc.convert(traditional_text)  # 进行转换

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(simplified_text)


input_file_path = "data/temporary/DRCD-backtrans/traditional.txt"  # 输入文件路径
output_file_path = "data/temporary/DRCD-backtrans/DRCD-backtrans.txt"  # 输出文件路径
convert_traditional_to_simplified(input_file_path, output_file_path)

print("DRCD-backtrans繁简转换完成！")
print('DRCD-backtrans.txt已格式化！')


"""
格式化ODSQA_spokenq_test
"""

# 读取保存内容的文本文件
with open('data/raw/ODSQA_spokenq_test.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()

# 使用正则表达式提取问题和内容
pattern = r'question: (.*?)\n.*?text: (.*?)\n'
matches = re.findall(pattern, input_text, re.DOTALL)

# 将匹配结果整理成纯文本格式
output_text = ""
for match in matches:
    output_text += f"{match[0]}\n{match[1]}\n"

# 将结果写入一个新的纯文本文件
with open('data/temporary/ODSQA_spokenq_test/traditional.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(output_text)

path = 'data/temporary/ODSQA_spokenq_test/traditional.txt'
dataFile = open(path, 'r', encoding='utf8')

qaList = dataFile.readlines()
total = []

# 逐行读取数据，并将每两行作为一个问答对
for i in range(0, len(qaList), 4):
    question = qaList[i].strip()
    answer = qaList[i + 1].strip()
    qa = [question, answer]
    total.append(qa)

# 关闭原始数据文件
dataFile.close()

resultData = open('data/temporary/ODSQA_spokenq_test/traditional.txt', 'w', encoding='utf8')

# 遍历处理后的问答对列表，写入格式化后的数据文件
for i in total:
    question = i[0].replace('"', '')
    answer = i[1].replace('"', '')
    qa = question + '|' + answer
    resultData.write(qa + '\n')
    # 关闭结果数据文件
resultData.close()

print('ODSQA_spokenq_test初步转换完成！')

def convert_traditional_to_simplified(input_path, output_path):
    cc = opencc.OpenCC("t2s")  # 创建一个转换器，从繁体转换为简体

    with open(input_path, "r", encoding="utf-8") as input_file:
        traditional_text = input_file.read()

    simplified_text = cc.convert(traditional_text)  # 进行转换

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(simplified_text)


input_file_path = "data/temporary/ODSQA_spokenq_test/traditional.txt"  # 输入文件路径
output_file_path = "data/temporary/ODSQA_spokenq_test/ODSQA_spokenq_test.txt"  # 输出文件路径
convert_traditional_to_simplified(input_file_path, output_file_path)

print("ODSQA_spokenq_test繁简转换完成！")
print('ODSQA_spokenq_test.txt已格式化！')

"""
格式化chatterbot-corpus-master
"""

path = 'data/raw/chatterbot-corpus-master.txt'
dataFile = open(path, 'r', encoding='utf8')

qaList = dataFile.readlines()
total = []

# 逐行读取数据，并将每两行作为一个问答对
for i in range(0, len(qaList), 4):
    question = qaList[i].strip()
    answer = qaList[i + 1].strip()
    qa = [question, answer]
    total.append(qa)

# 关闭原始数据文件
dataFile.close()

resultData = open('data/temporary/chatterbot-corpus-master/chatterbot-corpus-master.txt', 'w', encoding='utf8')

# 遍历处理后的问答对列表，写入格式化后的数据文件
for i in total:
    question = i[0].replace('-', '').replace(' ', '')
    answer = i[1].replace('-', '').replace(' ', '')
    qa = question + '|' + answer
    resultData.write(qa + '\n')
# 关闭结果数据文件
resultData.close()

print('chatterbot-corpus-master.txt已格式化！')

"""
合并两数据集
"""
# 读取 DRCD-backtrans.txt
with open('data/temporary/DRCD-backtrans/DRCD-backtrans.txt', 'r', encoding='utf-8') as drcd_file:
    drcd_text = drcd_file.read()

# 读取 DRCD-backtrans.txt
with open('data/temporary/ODSQA_spokenq_test/ODSQA_spokenq_test.txt', 'r', encoding='utf-8') as odsqa_file:
    odsqa_text = odsqa_file.read()

# 读取 chatterbot-corpus-master.txt
with open('data/temporary/chatterbot-corpus-master/chatterbot-corpus-master.txt', 'r', encoding='utf-8') as chat_file:
    chat_text = chat_file.read()

# 读取 xiaohuangji50w.txt
with open('data/raw/xiaohuangji50w.txt', 'r', encoding='utf-8') as xhj_file:
    xhj_text = xhj_file.read()
# 合并文本
merged_text = drcd_text + odsqa_text + chat_text + xhj_text

# 写入 corpus.txt
with open('data/standardlized/corpus.txt', 'w', encoding='utf-8') as corpus_file:
    corpus_file.write(merged_text)
print('数据集已全部格式化！')