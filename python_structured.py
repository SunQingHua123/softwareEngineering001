# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize

from nltk import wordpunct_tokenize
from io import StringIO
# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet

#---------------------定义正则表达式----------------------
# 匹配变量赋值的正则表达式
PATTERN_VAR_EQUAL = re.compile("(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
# 匹配for循环中的变量的正则表达式
PATTERN_VAR_FOR = re.compile("for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")
#---------------------定义正则表达式----------------------

#---------------------子函数1：修复程序输入输出------------
def repair_program_io(code):
    """
    修复程序输入输出

    该函数用于修复包含程序输入输出的代码块，包括Python交互式会话中的输入（In）、输出（Out）以及多行输入的情况。

    Args:
        code (str): 待修复的代码块字符串。

    Returns:
        tuple: 修复后的代码块字符串和代码块列表。

    示例：
        修复代码块中的输入输出标志：

        >>> code = "In [1]: x = 10\nOut [1]: 10\n"
        >>> repair_program_io(code)
        ('x = 10\n10\n', ['x = 10', '10'])
    """
    # 正则表达式模式用于识别输入输出标志
    pattern_case1_in = re.compile("In ?\[\d+]: ?")  # 输入标志
    pattern_case1_out = re.compile("Out ?\[\d+]: ?")  # 输出标志
    pattern_case1_cont = re.compile("( )+\.+: ?")  # 续行标志

    pattern_case2_in = re.compile(">>> ?")  # 输入标志（备选）
    pattern_case2_cont = re.compile("\.\.\. ?")  # 续行标志（备选）

    # 将所有模式组合到列表中以便处理
    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                pattern_case2_in, pattern_case2_cont]

    # 将代码块拆分成行
    lines = code.split("\n")
    lines_flags = [0 for _ in range(len(lines))]  # 标志行列表，用于标识每行的类型

    code_list = []  # 修复后的代码块列表

    # 匹配模式
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        for pattern_idx in range(len(patterns)):
            if re.match(patterns[pattern_idx], line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False  # 修复标志，用于指示是否成功修复了代码块

    # 修复代码块
    if lines_flags.count(0) == len(lines_flags):  # 无需修复的情况
        repaired_code = code
        code_list = [code]
        bool_repaired = True

    # 典型情况下，输入输出标志之间没有续行标志，因此可以直接删除标志行即可修复代码块
    elif re.match(re.compile("(0*1+3*2*0*)+"), lines_flags_string) or \
            re.match(re.compile("(0*4+5*0*)+"), lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        if lines_flags[0] == 0:
            flag = 0
            while (flag == 0):
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""  # 清空

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                # 清除子块记录
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        # 避免丢失最后一个单元
        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    # 非典型情况下，仅删除每个输出标志后的0标志行即可修复代码块
    if not bool_repaired:
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list
#---------------------子函数1：修复程序输入输出---------------------

#---------------------子函数2：获取AST中的变量名集合----------------
def get_vars(ast_root):
    """
    获取AST中的变量名集合

    该函数从给定的AST根节点中提取所有变量名，并返回一个按字母顺序排序的集合。

    Args:
        ast_root (ast.Module): AST的根节点。

    Returns:
        set: 按字母顺序排序的变量名集合。

    示例：
        从AST中提取变量名：

        >>> import ast
        >>> code = "x = 10\ny = 20\n"
        >>> ast_root = ast.parse(code)
        >>> get_vars(ast_root)
        {'x', 'y'}
    """
    return sorted(
        {node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})
#---------------------子函数2：获取AST中的变量名集合------------------

#---------------------子函数3：启发式方法获取代码中的变量名------------
def get_vars_heuristics(code):
    """
    启发式方法获取代码中的变量名

    该函数尝试使用启发式方法从给定的代码中提取变量名。首先，它尝试对代码进行解析，然后处理剩余的行。

    Args:
        code (str): 待处理的代码字符串。

    Returns:
        set: 变量名集合。

    示例：
        从代码中提取变量名：

        >>> code = "x = 10\nfor i in range(5):\n    y = i\n"
        >>> get_vars_heuristics(code)
        {'i', 'x', 'y'}
    """
    varnames = set()  # 存储变量名的集合
    code_lines = [_ for _ in code.split("\n") if len(_.strip())]  # 分割代码行并去除空行

    # 最佳尝试解析
    start = 0
    end = len(code_lines) - 1
    bool_success = False
    while not bool_success:
        try:
            root = ast.parse("\n".join(code_lines[start:end]))  # 尝试解析代码片段
        except:
            end -= 1
        else:
            bool_success = True
    varnames = varnames.union(set(get_vars(root)))  # 提取变量名并添加到集合中

    # 处理剩余的代码行
    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)  # 解析单行代码
        except:
            # 匹配 PATTERN_VAR_EQUAL
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]  # 移除 "="
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

            # 匹配 PATTERN_VAR_FOR
            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]  # 移除 "for" 和 "in"
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

        else:
            varnames = varnames.union(get_vars(root))  # 提取变量名并添加到集合中

    return varnames
#---------------------子函数3：启发式方法获取代码中的变量名------------

#---------------------子函数4：Python代码解析器----------------------
def PythonParser(code):
    """
    Python代码解析器

    该函数将给定的Python代码解析为标记序列。

    Args:
        code (str): 待解析的Python代码字符串。

    Returns:
        tuple: 一个包含以下元素的元组：
            - list: 标记化的代码序列。
            - bool: 变量解析失败的标志。
            - bool: 标记解析失败的标志。

    示例：
        解析Python代码为标记序列：

        >>> code = "x = 10\ny = 20\nprint(x + y)"
        >>> PythonParser(code)
        (['x', '=', '10', 'y', '=', '20', 'VAR', '+', 'VAR', 'VAR'], False, False)
    """
    bool_failed_var = False  # 变量解析失败的标志
    bool_failed_token = False  # 标记解析失败的标志

    try:
        root = ast.parse(code)  # 尝试解析代码为AST
        varnames = set(get_vars(root))  # 获取AST中的变量名集合
    except:
        # 如果解析失败，尝试修复代码并重新解析
        repaired_code, _ = repair_program_io(code)
        try:
            root = ast.parse(repaired_code)
            varnames = set(get_vars(root))
        except:
            bool_failed_var = True
            # 如果解析仍然失败，则使用启发式方法提取变量名
            varnames = get_vars_heuristics(code)

    tokenized_code = []  # 存储标记化的代码序列

    # 第一次尝试解析代码，以检查是否存在语法错误
    def first_trial(_code):
        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            term = next(g)
        except:
            return False
        else:
            return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)
    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False  # 解析完成的标志
    while not bool_finished:
        term_type = term[0]  # 标记类型
        lineno = term[2][0] - 1  # 行号
        posno = term[3][1] - 1  # 列号
        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        # 获取下一个标记
        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                bool_failed_token = True
                code_lines = code.split("\n")
                if lineno > len(code_lines) - 1:
                    print(sys.exc_info())
                else:
                    failed_code_line = code_lines[lineno]
                    if posno < len(failed_code_line) - 1:
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(failed_code_line)
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token
#---------------------子函数4：Python代码解析器----------------------

#---------------------子函数5：恢复缩略词的全称----------------------
def revert_abbrev(line):
    """
    恢复缩略词的全称

    该函数将缩略词还原为其全称。

    Args:
        line (str): 待处理的字符串。

    Returns:
        str: 还原缩略词后的字符串。

    示例：
        >>> revert_abbrev("She's here.")
        'She is here.'
    """
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)  # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")  # 's
    pat_s2 = re.compile("(?<=s)\"s?")  # s
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")  # not
    pat_would = re.compile("(?<=[a-zA-Z])\"d")  # would
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")  # will
    pat_am = re.compile("(?<=[I|i])\"m")  # am
    pat_are = re.compile("(?<=[a-zA-Z])\"re")  # are
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")  # have

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line
#---------------------子函数5：恢复缩略词的全称----------------------

#---------------------子函数6：获取词性-----------------------------
def get_wordpos(tag):
    """
    获取词性对应的WordNet标签

    根据词性标签返回相应的WordNet标签，用于词性还原。

    Args:
        tag (str): Penn Treebank 词性标签。

    Returns:
        str: 对应的WordNet标签。
    """
    # 根据词性标签返回相应的WordNet标签
    if tag.startswith('J'):  # 形容词
        return wordnet.ADJ
    elif tag.startswith('V'):  # 动词
        return wordnet.VERB
    elif tag.startswith('N'):  # 名词
        return wordnet.NOUN
    elif tag.startswith('R'):  # 副词
        return wordnet.ADV
    else:
        return None
#---------------------子函数6：获取词性-----------------------------

#---------------------子函数7：句子的去冗---------------------------
def process_nl_line(line):
    """
    对句子进行去冗处理。

    对输入的句子进行去冗处理，包括缩写还原、制表符替换、多余换行符去除、句子拼接、多余空格去除、去除括号内内容、去除首尾空格以及骆驼命名转下划线。

    Args:
        line (str): 待处理的句子。

    Returns:
        str: 去冗处理后的句子。
    """
    # 缩写还原
    line = revert_abbrev(line)
    # 制表符替换
    line = re.sub('\t+', '\t', line)
    # 多余换行符去除
    line = re.sub('\n+', '\n', line)
    # 句子拼接
    line = line.replace('\n', ' ')
    # 多余空格去除
    line = re.sub(' +', ' ', line)
    # 去除首尾空格
    line = line.strip()
    # 去除括号内内容
    space = re.compile(r"\([^(|^)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除首尾空格
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    return line
#---------------------子函数7：句子的去冗---------------------------

#---------------------子函数8：句子的分词---------------------------
def process_sent_word(line):
    """
    对句子进行分词处理。

    对输入的句子进行分词处理，包括单词提取、小数、字符串、十六进制、数字和字符的替换，全词小写化，词性标注，词性还原和词干提取。

    Args:
        line (str): 待处理的句子。

    Returns:
        list: 分词后的单词列表。
    """
    # 找单词
    line = re.findall(r"\w+|[^\s\w]", line)
    line = ' '.join(line)
    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符
    other = re.compile(r"(?<![A-Z|a-z_])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words = line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list
#---------------------子函数8：句子的分词---------------------------

#---------------------子函数9：字符过滤（所有）---------------------
def filter_all_invachar(line):
    """
    去除句子中的非常用符号，防止解析错误。

    对输入的句子进行处理，去除非常用符号，包括除了数字、字母、中划线、下划线、单引号和双引号之外的字符。

    Args:
        line (str): 待处理的句子。

    Returns:
        str: 去除非常用符号后的句子。
    """
    assert isinstance(line, object)
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line
#---------------------子函数9：字符过滤（所有）---------------------

#---------------------子函数10：字符过滤（部分）--------------------
def filter_part_invachar(line):
    """
    去除句子中的非常用符号，防止解析错误。

    对输入的句子进行处理，去除非常用符号，包括除了数字、字母、中划线、下划线、单引号和双引号之外的字符。

    Args:
        line (str): 待处理的句子。

    Returns:
        str: 去除非常用符号后的句子。
    """
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line
#---------------------子函数10：字符过滤（部分）---------------------

#-----------------------主函数：代码的tokens------------------------
def python_code_parse(line):
    """
    解析Python代码的tokens。

    对输入的Python代码进行解析，获取其tokens。

    Args:
        line (str): 待解析的Python代码。

    Returns:
        list: Python代码的tokens列表，如果解析失败则返回'-1000'。
    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)  # 新增加
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower() for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        return token_list
        # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'
#-----------------------主函数：代码的tokens------------------------

#-----------------------主函数：句子的tokens------------------------
def python_query_parse(line):
    """
    解析Python查询的句子tokens。

    对输入的Python查询句子进行解析，获取其tokens。

    Args:
        line (str): 待解析的Python查询句子。

    Returns:
        list: Python查询句子的tokens列表。

    示例：
        >>> python_query_parse("加载pickle文件")
        ['加载', 'pickle', '文件']
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉括号
    for i in range(0, len(word_list)):
        if re.findall('[()]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list


def python_context_parse(line):
    """
    解析Python上下文的句子tokens。

    对输入的Python上下文句子进行解析，获取其tokens。

    Args:
        line (str): 待解析的Python上下文句子。

    Returns:
        list: Python上下文句子的tokens列表。

    示例：
        >>> python_context_parse("加载pickle文件")
        ['加载', 'pickle', '文件']
    """
    line = filter_part_invachar(line)
    # 在这一步的时候驼峰命名被转换成了下划线
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list
#-----------------------主函数：句子的tokens-----------------------


if __name__ == '__main__':
    # 解析并打印给定的Python查询句子
    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    print(python_query_parse(
        'python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    # 解析并打印给定的Python上下文句子
    print(python_context_parse(
        'How to calculate the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... + n^2 until a user specified sum has been reached sql()'))
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    # 解析并打印给定的Python代码片段
    print(python_code_parse(
        'if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    print(python_code_parse(
        "diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    # 解析并打印更多的Python代码片段
    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print(key, 'corresponds to', d[key])"))
    print(python_code_parse(
        '  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
