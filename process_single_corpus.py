import pickle  
from collections import Counter  

def load_pickle(filename):
    """加载pickle文件

    Args:
        filename (str): pickle文件路径

    Returns:
        data (object): 从pickle文件中加载的数据
    """
    with open(filename, 'rb') as f:  # 以二进制读取模式打开文件
        data = pickle.load(f, encoding='iso-8859-1')  # 使用pickle加载数据，并指定编码为'iso-8859-1'
    return data  # 返回加载的数据

def split_data(total_data, qids):
    """将数据按单一问题和多问题分割

    Args:
        total_data (list): 全部数据列表
        qids (list): 问题ID列表

    Returns:
        tuple: 单一问题数据列表，多问题数据列表
    """
    result = Counter(qids)  # 使用Counter统计每个问题ID出现的次数
    total_data_single = []  # 存储单一问题的数据
    total_data_multiple = []  # 存储多问题的数据

    for data in total_data:  # 遍历所有数据
        if result[data[0][0]] == 1:  # 如果问题ID只出现一次
            total_data_single.append(data)  # 添加到单一问题数据列表
        else:  # 如果问题ID出现多次
            total_data_multiple.append(data)  # 添加到多问题数据列表
    
    return total_data_single, total_data_multiple  # 返回分割后的数据列表

def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """处理staqc数据并按单一问题和多问题保存

    Args:
        filepath (str): 输入数据文件路径
        save_single_path (str): 保存单一问题数据的文件路径
        save_multiple_path (str): 保存多问题数据的文件路径
    """
    with open(filepath, 'r') as f:  # 以读取模式打开文件
        total_data = eval(f.read())  # 读取文件内容并使用eval解析为Python对象
    qids = [data[0][0] for data in total_data]  # 提取每条数据的第一个问题ID
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, "w") as f:  # 以写模式打开单一问题数据保存路径
        f.write(str(total_data_single))  # 将单一问题数据写入文件
    with open(save_multiple_path, "w") as f:  # 以写模式打开多问题数据保存路径
        f.write(str(total_data_multiple))  # 将多问题数据写入文件

def data_large_processing(filepath, save_single_path, save_multiple_path):
    """处理大型数据并按单一问题和多问题保存

    Args:
        filepath (str): 输入数据文件路径
        save_single_path (str): 保存单一问题数据的文件路径
        save_multiple_path (str): 保存多问题数据的文件路径
    """
    total_data = load_pickle(filepath)  # 加载pickle文件中的数据
    qids = [data[0][0] for data in total_data]  # 提取每条数据的第一个问题ID
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, 'wb') as f:  # 以二进制写模式打开单一问题数据保存路径
        pickle.dump(total_data_single, f)  # 将单一问题数据写入pickle文件
    with open(save_multiple_path, 'wb') as f:  # 以二进制写模式打开多问题数据保存路径
        pickle.dump(total_data_multiple, f)  # 将多问题数据写入pickle文件

def single_unlabeled_to_labeled(input_path, output_path):
    """将单一未标记数据转换为标记数据并保存

    Args:
        input_path (str): 输入未标记数据文件路径
        output_path (str): 输出标记数据文件路径
    """
    total_data = load_pickle(input_path)  # 加载pickle文件中的数据
    labels = [[data[0], 1] for data in total_data]  # 为每条数据添加标签1
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 对数据按问题ID和标签排序

    with open(output_path, "w") as f:  # 以写模式打开输出文件
        f.write(str(total_data_sort))  # 将标记数据写入文件

if __name__ == "__main__":
    # STAQC Python数据处理
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'  # 输入数据文件路径
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'  # 单一问题数据保存路径
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'  # 多问题数据保存路径
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)  # 处理并保存数据

    # STAQC SQL数据处理
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'  # 输入数据文件路径
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'  # 单一问题数据保存路径
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'  # 多问题数据保存路径
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)  # 处理并保存数据

    # 大型Python数据处理
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'  # 输入数据文件路径
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'  # 单一问题数据保存路径
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'  # 多问题数据保存路径
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)  # 处理并保存数据

    # 大型SQL数据处理
    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'  # 输入数据文件路径
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'  # 单一问题数据保存路径
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'  # 多问题数据保存路径
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)  # 处理并保存数据

    # 将单一未标记数据转换为标记数据
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'  # 标记数据保存路径
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'  # 标记数据保存路径
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)  # 转换并保存标记数据
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)  # 转换并保存标记数据
