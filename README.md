# **软件工程实验**

[TOC]

## 一、项目概述及文件结构

### 1.1 项目概述

本项目是NLP任务的数据预处理阶段。通过所给的python文件，对文件进行代码代码规范，主要进行注释工作，增强代码可读性。

### 1.2 文件结构

```
├── data_preprocessing（modify）  
│   └── embaddings_process.py  
│   └── getStru2Vec.py
│   └── process_single_corpus.py
│   └── python_structured.py
│   └── sqlang_structured.py
│   └── word_dirt.py

```

------



## 二、项目文件说明

### 2.1 process_single_corpus.py

#### 2.1.1 文件概述

`process_single_corpus.py`脚本用于处理和分割各种语料库中的数据，并将其分为单一问题数据和多问题数据。它还可以将未标记的数据转换为标记数据并保存处理结果。

#### 2.1.2 依赖库

- `pickle`: 用于序列化和反序列化Python对象，主要用于加载和保存数据文件。
- `collections.Counter`: 用于统计数据中各个元素的出现次数。

#### 2.1.3 类和方法说明

- **`load_pickle`**: 从pickle文件中加载数据并返回。
- **`split_data`**: 将数据按单一问题和多问题分割。该函数统计每个问题ID出现的次数，并根据出现次数将数据分为单一问题数据列表和多问题数据列表。
- **`data_staqc_processing`**: 处理STAQC数据并按单一问题和多问题保存。该函数从文件中读取数据，调用`split_data()`函数分割数据，并将结果分别保存到指定的文件路径中。
- **`data_large_processing`**: 处理大型数据并按单一问题和多问题保存。该函数从pickle文件中加载数据，调用`split_data()`函数分割数据，并将结果分别保存到指定的文件路径中。
- **`single_unlabeled_to_labeled`**: 将单一未标记数据转换为标记数据并保存。该函数从pickle文件中加载数据，为每条数据添加标签1，对数据进行排序，并将结果保存到指定的文件路径中。

------

### 2.2 word_dirt.py文件

#### 2.2.1 文件概述

这个Python脚本用于处理和提取语料库中的词汇表，并保存处理后的词汇表。具体来说，它从两个语料库中提取词汇，排除不需要的词汇，并将最终的词汇表保存到指定文件中。

#### 2.2.2 依赖库

- `pickle`: 用于序列化和反序列化Python对象。它在脚本中被用来加载和保存pickle文件的数据。
- `eval`: 虽然不是一个导入的库，但它被用来将字符串表示的Python对象转换为实际的Python对象。

#### 2.2.3 类和方法说明

- **`get_vocab`**: 根据给定的两个语料库，获取词汇表。该函数遍历语料库中的数据，并将所有单词添加到一个集合中，最终返回词汇表。

- **`load_pickle`**: 从pickle文件中加载数据并返回。

- **`vocab_processing`**: 用于处理语料库文件和保存词汇表的文件路径。该函数调用`load_pickle()`函数加载语料库数据，然后调用`get_vocab()`函数获取词汇表，并将词汇表保存到指定的文件路径中。

- **`final_vocab_processing`**: 首先从文件中加载已有的词汇表，然后调用`get_vocab()`函数获取新的词汇表。将新的词汇表与已有词汇表进行比较，找到新的单词，并将其保存到指定的文件路径中。

------

### 2.3 python_structured.py文件

#### 2.3.1 文件概述
解析 Python 代码，修复代码中的变量命名问题； 代码重构，添加变量名的注释。

#### 2.3.2 依赖库

- `re`：用于正则表达式匹配和替换
- `ast`:用于处理Python代码抽象语法树
- `sys`:用于程序与解释器交互
- `token`和 `tokenize`：用于解析 Python 代码中的 token
- `io.StringIO`：用于在内存中操作字符串作为文件
- `inflection`：用于进行单词的单复数转换
- `nltk`：自然语言处理工具包，用于词性标注、分词和词形还原

#### 2.3.3 类和方法说明

##### 类：PythonParser
该类包含了一系列用于解析Python代码的方法。

##### 方法：

- **`get_vars(ast_root)`**：获取变量名。
- **`PythonParser(code)`**: 将代码字符串解析为Token 序列，并且执行变量解析。 
- **`first_trial(_code)`**:尝试将该代码字符串解析为token令牌序列。

- **`repair_program_io(code)`**: 修复包含程序输入输出的代码块，返回修复后的代码块字符串和代码块列表。
- **`get_vars_heuristics(code)`**: 使用启发式方法从给定的代码中提取变量名，并返回一个按字母顺序排序的集合。
- **`revert_abbrev(sentence)`**: 将缩略词还原为其全称。
- **`get_wordpos(tag)`**: 根据词性标签返回相应的WordNet标签。
- **`process_nl_line(sentence)`**: 对输入的句子进行去冗处理。
- **`process_sent_word(sentence)`**: 对输入的句子进行分词处理。
- **`filter_all_invachar(sentence)`**: 去除句子中的非常用符号。
- **`filter_part_invachar(sentence)`**: 去除句子中的非常用符号。

------

### 2.4 sql_structured.py文件

#### 2.4.1 文件概述：

该文件包含了用于解析 SQL 查询语句和自然语言句子的代码。主要实现了两个功能：SQL 查询语句解析和自然语言句子处理。通过这些功能，可以将输入的 SQL 查询语句或自然语言句子转换为标记化的单词列表，便于后续处理和分析。

#### 2.4.2 依赖库：

- `sqlparse`：用于解析 SQL 查询语句。
- `inflection`：用于将骆驼命名法转换为下划线命名法。
- `NLTK`：用于自然语言处理，包括词性标注、词性还原和词干提取。

#### 2.4.3 类和方法说明：

- **`SqlangParser`** 类：
    - **`sanitizeSql(sql)`** 方法：清理和规范化 SQL 语句。
    - **`parseStrings(tok)`** 方法：解析字符串标记，将其替换为预定义的字符串或正则表达式标记。
    - **`renameIdentifiers(tok)`** 方法：重命名标识符，包括列名和表名，并替换数值类型为固定标记。
    - **`identifySubQueries(tokenList)`** 方法：识别解析树中的子查询。
    - **`identifyLiterals(tokenList)`** 方法：识别解析树中的字面量和标识符。
    - **`identifyFunctions(tokenList)`** 方法：识别解析树中的函数。
    - **`identifyTables(tokenList)`** 方法：识别解析树中的表名。
    - **`removeWhitespaces(tok)`** 方法：移除解析树中的空白字符。
    - **`getTokens(parse)`** 方法：从解析树中获取所有标记。
    - **`parseSql()`** 方法：获取解析结果的标记列表。
- **其他辅助函数**：
    - **`tokenizeRegex(s)`**：使用预定义的正则表达式扫描器对字符串进行标记。
    - **`revert_abbrev(line)`**：处理缩略词，将缩写形式转换为完整形式。
    - **`get_wordpos(tag)`**：根据词性标签获取对应的 WordNet 词性。
    - **`process_nl_line(line)`**：对自然语言句子进行预处理。
    - **`process_sent_word(line)`**：对句子进行分词和词性标注处理。
    - **`filter_all_invachar(line)`**：过滤句子中的所有非常用字符。
    - **`filter_part_invachar(line)`**：过滤句子中的部分非常用字符。
    - **`sqlang_code_parse(line)`**：解析 SQL 代码并返回标记后的单词列表。
    - **`sqlang_query_parse(line)`**：解析 SQL 查询语句并返回标记后的单词列表。

------

### 2.5 getStru2Vec.py文件

#### 2.5.1 文件概述

这个文件实现了一个多进程解析器，用于处理Python代码和SQL查询语句。主要功能包括对Python和SQL数据进行解析，并将解析结果保存到文件中。

#### 2.5.2 依赖库

- `pickle`: 用于序列化和反序列化Python对象。
- `multiprocessing`: 用于创建并行进程。
- `python_structured` 和 `sqlang_structured`: 自定义模块，包含了对Python和SQL语言的结构化解析功能。

#### 2.5.3 类和方法说明

- **`multipro_python_query(data_list)`**: 使用多进程对Python查询数据进行处理。
- **`multipro_python_code(data_list)`**: 使用多进程对Python代码进行处理。
- **`multipro_python_context(data_list)`**: 使用多进程对Python上下文进行处理。
- **`multipro_sqlang_query(data_list)`**: 使用多进程对SQLang查询数据进行处理。
- **`multipro_sqlang_code(data_list)`**: 并行处理SQLang代码解析。
- **`multipro_sqlang_context(data_list)`**: 并行处理SQLang上下文解析。
- **`parse(data_list, split_num, context_func, query_func, code_func)`**: 并行解析SQLang数据。
- **`main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func)`**: 主函数，解析数据并保存。

------

### 2.6 embeddings_process.py文件

#### 2.6.1 文件概述

这个文件实现了一些用于处理词向量文件和构建词典的功能，同时提供了一些辅助函数用于将文本序列化。

#### 2.6.2 依赖库

- `pickle`: 用于序列化和反序列化Python对象。
- `numpy`: 用于数值计算。
- `gensim.models.KeyedVectors`: 用于加载和处理词向量模型。

#### 2.6.3 类和方法说明

- **`trans_bin(path1, path2)`**: 将文本形式的词向量文件转换为二进制文件。
- **`get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path)`**: 构建新的词典和词向量矩阵。
- **`get_index(type, text, word_dict)`**: 获取文本中每个词在词典中的位置。
- **`serialization(word_dict_path, type_path, final_type_path)`**: 将训练、测试、验证语料序列化。



### 三、注释格式

#### 3.1 方法注释

```
def function_name(arg1, arg2, ...):
    """
    描述方法功能和作用

    Args:
        arg1 (type): 参数1的描述
        arg2 (type): 参数2的描述
        ...: 其他参数的描述

    Returns:
        type1: 返回值的描述
        type2: 返回值的描述
    """
    # 函数体

```

####  3.2 语句注释

**示例：**

```
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
```



#### 3.3 长文件代码分块注释

**示例：**

```
#---------------------子函数4：获取词性----------------------
def get_wordpos(tag):
    """根据词性标签获取对应的WordNet词性

    Args:
        tag (str): 词性标签

    Returns:
        str: 对应的WordNet词性，如果没有对应的词性，返回 None
    """
    if tag.startswith('J'):
        return wordnet.ADJ  # 形容词
    elif tag.startswith('V'):
        return wordnet.VERB  # 动词
    elif tag.startswith('N'):
        return wordnet.NOUN  # 名词
    elif tag.startswith('R'):
        return wordnet.ADV  # 副词
    else:
        return None  # 其他情况返回 None
#---------------------子函数4：获取词性----------------------
```

