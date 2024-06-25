import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *

def multipro_python_query(data_list):
    """使用多进程对Python查询数据进行处理

    Args:
        data_list (list): 包含Python查询数据的列表

    Returns:
        list: 包含处理后的Python查询数据的列表
    """
    return [python_query_parse(line) for line in data_list]

def multipro_python_code(data_list):
    """使用多进程对Python代码进行处理

    Args:
        data_list (list): 包含Python代码的列表

    Returns:
        list: 包含处理后的Python代码的列表
    """
    return [python_code_parse(line) for line in data_list]

def multipro_python_context(data_list):
    """使用多进程对Python上下文进行处理

    Args:
        data_list (list): 包含Python上下文的列表

    Returns:
        list: 包含处理后的Python上下文的列表
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result

def multipro_sqlang_query(data_list):
    """使用多进程对SQLang查询数据进行处理

    Args:
        data_list (list): 包含SQLang查询数据的列表

    Returns:
        list: 包含处理后的SQLang查询数据的列表
    """
    return [sqlang_query_parse(line) for line in data_list]


def multipro_sqlang_code(data_list):
    """并行处理SQLang代码解析

    Args:
        data_list (list): 包含SQLang代码的字符串列表

    Returns:
        list: SQLang代码的tokens列表
    """
    return [sqlang_code_parse(line) for line in data_list]

def multipro_sqlang_context(data_list):
    """并行处理SQLang上下文解析

    Args:
        data_list (list): 包含SQLang上下文的字符串列表

    Returns:
        list: SQLang上下文的tokens列表
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

def parse(data_list, split_num, context_func, query_func, code_func):
    """并行解析SQLang数据

    Args:
        data_list (list): 待解析的数据列表
        split_num (int): 切分的数量
        context_func (function): 上下文解析函数
        query_func (function): 查询解析函数
        code_func (function): 代码解析函数

    Returns:
        tuple: 包含上下文、查询和代码解析结果的元组
    """
    pool = multiprocessing.Pool()
    # 将数据列表分割为多个子列表，以便并行处理
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    # 并行处理上下文解析函数
    results = pool.map(context_func, split_list)
    # 合并结果列表
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    # 并行处理查询解析函数
    results = pool.map(query_func, split_list)
    # 合并结果列表
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    # 并行处理代码解析函数
    results = pool.map(code_func, split_list)
    # 合并结果列表
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    pool.close()
    pool.join()

    return context_data, query_data, code_data


def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    """主函数：解析数据并保存

    Args:
        lang_type (str): 数据类型 ('python' 或 'sql')
        split_num (int): 切分的数量
        source_path (str): 输入数据文件路径
        save_path (str): 输出数据文件路径
        context_func (function): 上下文解析函数
        query_func (function): 查询解析函数
        code_func (function): 代码解析函数
    """
    # 从文件中加载数据
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 并行解析数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)

    # 获取问题ID列表
    qids = [item[0] for item in corpus_lis]

    # 构建总数据列表
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 将解析结果保存到文件
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

# 定义常量
python_type = 'python'
sqlang_type = 'sql'
words_top = 100
split_num = 1000

if __name__ == '__main__':
    # 定义STaqc数据的Python路径和保存路径
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    # 定义STaqc数据的SQL路径和保存路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 处理STaqc数据的Python部分
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    # 处理STaqc数据的SQL部分
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    # 定义大型语料的Python路径和保存路径
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    # 定义大型语料的SQL路径和保存路径
    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    # 处理大型语料的Python部分
    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    # 处理大型语料的SQL部分
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)
