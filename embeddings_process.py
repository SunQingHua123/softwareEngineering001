import pickle
import numpy as np
from gensim.models import KeyedVectors

def trans_bin(path1, path2):
    """
    将文本形式的词向量文件转换为二进制文件

    Args:
        path1 (str): 原始词向量文件的路径
        path2 (str): 转换后的二进制词向量文件的路径
    """
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)

def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    """
    构建新的词典和词向量矩阵

    Args:
        type_vec_path (str): 转换后的二进制词向量文件的路径
        type_word_path (str): 词汇表文件的路径
        final_vec_path (str): 输出的词向量矩阵文件的路径
        final_word_path (str): 输出的词汇表文件的路径
    """
    # 加载词向量文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    # 加载词汇表文件
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())

    # 初始化词典和词向量矩阵
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 其中0 PAD_ID, 1 SOS_ID, 2 EOS_ID, 3 UNK_ID
    word_vectors = [np.zeros(shape=(1, 300)).squeeze()] * 4

    # 遍历词汇表中的词语
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])  # 加载词向量
            word_dict.append(word)
        except KeyError:
            pass

    # 转换为NumPy数组
    word_vectors = np.array(word_vectors)

    # 构建词汇表的反向映射
    word_dict = dict(map(reversed, enumerate(word_dict)))

    # 保存词向量矩阵文件
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    # 保存词汇表文件
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")

def get_index(type, text, word_dict):
    """
    获取文本中每个词在词典中的位置

    Args:
        type (str): 文本类型，'code' 或其他
        text (list): 文本列表，包含单词或标记的列表
        word_dict (dict): 词典，将词映射到索引的字典

    Returns:
        list: 包含每个词在词典中的位置的列表

    Raises:
        KeyError: 如果词典中不存在某个单词，则会引发 KeyError
    """
    location = []  # 用于存储每个词在词典中的位置

    if type == 'code':  # 如果文本类型是代码
        location.append(1)  # 将代码起始标记添加到位置列表中
        len_c = len(text)  # 获取文本的长度
        
        # 如果代码长度小于 350
        if len_c + 1 < 350:
            # 如果文本长度为 1 且内容为 '-1000'（表示为空）
            if len_c == 1 and text[0] == '-1000':
                location.append(2)  # 将代码结束标记添加到位置列表中
            else:
                # 遍历文本中的每个词
                for i in range(0, len_c):
                    # 获取词在词典中的索引，如果词不在词典中，则使用 UNK（未知）的索引
                    index = word_dict.get(text[i], word_dict['UNK'])
                    location.append(index)  # 将索引添加到位置列表中
                location.append(2)  # 将代码结束标记添加到位置列表中
        else:  # 如果代码长度大于等于 350
            # 仅获取前 348 个词的索引，然后添加代码结束标记
            for i in range(0, 348):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)
            location.append(2)
    else:  # 如果文本类型不是代码
        # 如果文本为空列表或者第一个元素为 '-10000'（表示为空）
        if len(text) == 0 or text[0] == '-10000':
            location.append(0)  # 将空标记的索引添加到位置列表中
        else:
            # 遍历文本中的每个词，获取其在词典中的索引
            for i in range(0, len(text)):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)  # 将索引添加到位置列表中

    return location  # 返回位置列表

def serialization(word_dict_path, type_path, final_type_path):
    """
    将训练、测试、验证语料序列化

    Args:
        word_dict_path (str): 包含词典的文件路径
        type_path (str): 包含语料的文件路径
        final_type_path (str): 序列化后保存的文件路径
    """
    # 加载词典文件
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    # 加载语料文件
    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    # 遍历每条语料
    for i in range(len(corpus)):
        qid = corpus[i][0]  # 获取语料的问题 ID

        # 获取查询的词在词典中的索引序列
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        # 获取代码的词在词典中的索引序列
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)
        # 获取查询的词在词典中的索引序列
        query_word_list = get_index('text', corpus[i][3], word_dict)
        block_length = 4  # 每条语料的块长度
        label = 0  # 每条语料的标签

        # 将查询的词序列长度截断或填充为 25
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (100 - len(Si1_word_list))
        # 将代码的词序列长度截断或填充为 350
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        # 将查询的词序列长度截断或填充为 25
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (25 - len(query_word_list))

        # 组装每条语料的数据
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    # 将序列化后的数据保存到文件
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)

if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # 最初基于Staqc的词典和词向量路径
    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # SQL 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # SQL 最后的词典和对应的词向量路径
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    # Python 待处理语料地址
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # Python 最后的词典和对应的词向量路径
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # 生成最终的词典和词向量
    get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # 生成 SQL 最后的词典和对应的词向量
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)

    # 生成 Python 最后的词典和对应的词向量
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)

    # 处理成打标签的形式并序列化
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')


