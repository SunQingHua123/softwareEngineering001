import pickle

def get_vocab(corpus1, corpus2):
    """从两个语料库中提取词汇表

    Args:
        corpus1 (list): 第一个语料库
        corpus2 (list): 第二个语料库

    Returns:
        set: 提取的词汇表集合
    """
    word_vocab = set()  # 初始化一个空的集合用于存储词汇

    for corpus in [corpus1, corpus2]:  # 遍历两个语料库
        for i in range(len(corpus)):  # 遍历每个语料
            word_vocab.update(corpus[i][1][0])  # 更新词汇集合，添加当前语料的第1部分的第0项
            word_vocab.update(corpus[i][1][1])  # 更新词汇集合，添加当前语料的第1部分的第1项
            word_vocab.update(corpus[i][2][0])  # 更新词汇集合，添加当前语料的第2部分的第0项
            word_vocab.update(corpus[i][3])  # 更新词汇集合，添加当前语料的第3部分

    print(len(word_vocab))  # 打印词汇集合的大小
    return word_vocab  # 返回词汇集合

def load_pickle(filename):
    """加载pickle文件

    Args:
        filename (str): pickle文件路径

    Returns:
        data (object): 从pickle文件中加载的数据
    """
    with open(filename, 'rb') as f:  # 以二进制读取模式打开文件
        data = pickle.load(f)  # 使用pickle加载数据
    return data  # 返回加载的数据

def vocab_processing(filepath1, filepath2, save_path):
    """处理词汇表并保存结果

    Args:
        filepath1 (str): 第一个数据文件路径
        filepath2 (str): 第二个数据文件路径
        save_path (str): 词汇表保存路径
    """
    with open(filepath1, 'r') as f:  # 以读取模式打开第一个文件
        total_data1 = set(eval(f.read()))  # 使用eval读取并解析文件内容为集合

    with open(filepath2, 'r') as f:  # 以读取模式打开第二个文件
        total_data2 = eval(f.read())  # 使用eval读取并解析文件内容

    word_set = get_vocab(total_data2, total_data2)  # 从第二个数据文件中提取词汇表

    excluded_words = total_data1.intersection(word_set)  # 计算两个词汇表的交集
    word_set = word_set - excluded_words  # 从词汇表中移除交集部分

    print(len(total_data1))  # 打印第一个词汇表的大小
    print(len(word_set))  # 打印处理后的词汇表大小

    with open(save_path, 'w') as f:  # 以写模式打开保存路径
        f.write(str(word_set))  # 将词汇表写入文件

if __name__ == "__main__":
    # 定义文件路径
    python_hnn = './data/python_hnn_data_teacher.txt'  # Python HNN数据文件路径
    python_staqc = './data/staqc/python_staqc_data.txt'  # Python STAQC数据文件路径
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'  # Python词汇表保存路径

    sql_hnn = './data/sql_hnn_data_teacher.txt'  # SQL HNN数据文件路径
    sql_staqc = './data/staqc/sql_staqc_data.txt'  # SQL STAQC数据文件路径
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'  # SQL词汇表保存路径

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'  # 新SQL STAQC数据文件路径
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'  # 大型SQL数据文件路径
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'  # 大型SQL词汇表保存路径

    vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)  # 处理词汇表并保存结果
