# -*- coding: utf-8 -*-
import re
import sqlparse 

# 骆驼命名法转换库
import inflection

# 自然语言处理库，进行词性标注和词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

# 词干提取库
from nltk.corpus import wordnet

#---------------------定义各种类型的常量---------------------
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13

# 类型字典，将常量映射为字符串表示
ttypes = {
    0: "OTHER", 
    1: "FUNCTION", 
    2: "BLANK", 
    3: "KEYWORD", 
    4: "INTERNAL", 
    5: "TABLE", 
    6: "COLUMN", 
    7: "INTEGER",
    8: "FLOAT", 
    9: "HEX", 
    10: "STRING", 
    11: "WILDCARD", 
    12: "SUBQUERY", 
    13: "DUD", 
}

# 定义一个正则表达式扫描器，用于识别特定模式
scanner = re.Scanner([
    (r"\[[^\]]*\]", lambda scanner, token: token),  # 匹配方括号中的内容
    (r"\+", lambda scanner, token: "REGPLU"),  # 匹配加号
    (r"\*", lambda scanner, token: "REGAST"),  # 匹配星号
    (r"%", lambda scanner, token: "REGCOL"),  # 匹配百分号
    (r"\^", lambda scanner, token: "REGSTA"),  # 匹配插入符号
    (r"\$", lambda scanner, token: "REGEND"),  # 匹配美元符号
    (r"\?", lambda scanner, token: "REGQUE"),  # 匹配问号
    (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),  # 匹配各种字符
    (r'.', lambda scanner, token: None),  # 匹配任意字符，但忽略它
])
#---------------------定义各种类型的常量---------------------

#---------------------子函数1：代码的规则--------------------
def tokenizeRegex(s):
    """使用预定义的正则表达式扫描器对字符串进行标记

    Args:
        s (str): 输入的字符串

    Returns:
        list: 标记后的字符串列表
    """
    results = scanner.scan(s)[0]  # 使用扫描器对输入字符串进行扫描
    return results  # 返回标记结果

#---------------------子函数2：代码的规则--------------------
class SqlangParser():
    @staticmethod
    def sanitizeSql(sql):
        """清理和规范化SQL语句

        Args:
            sql (str): 输入的SQL语句

        Returns:
            str: 清理后的SQL语句
        """
        s = sql.strip().lower()  # 去除首尾空白并转换为小写
        if not s[-1] == ";":
            s += ';'  # 如果SQL语句末尾没有分号，则添加分号
        s = re.sub(r'\(', r' ( ', s)  # 在左括号前后添加空格
        s = re.sub(r'\)', r' ) ', s)  # 在右括号前后添加空格
        words = ['index', 'table', 'day', 'year', 'user', 'text']  # 需要处理的保留字
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)  # 替换结尾的保留字
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)  # 替换中间的保留字
        s = s.replace('#', '')  # 移除井号
        return s  # 返回清理后的SQL语句

    def parseStrings(self, tok):
        """解析字符串标记

        Args:
            tok (sqlparse.sql.Token or sqlparse.sql.TokenList): 输入的标记

        处理字符串类型的标记，将其替换为预定义的字符串或正则表达式标记
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)  # 递归处理TokenList中的每个Token
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))  # 使用正则表达式标记字符串
            else:
                tok.value = "CODSTR"  # 将字符串替换为固定标记

    def renameIdentifiers(self, tok):
        """重命名标识符

        Args:
            tok (sqlparse.sql.Token or sqlparse.sql.TokenList): 输入的标记

        重命名列名和表名，替换数值类型为固定标记
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)  # 递归处理TokenList中的每个Token
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])  # 生成新的列名
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]  # 替换列名
        elif tok.ttype == TABLE:
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])  # 生成新的表名
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]  # 替换表名
        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"  # 将浮点数替换为固定标记
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"  # 将整数替换为固定标记
        elif tok.ttype == HEX:
            tok.value = "CODHEX"  # 将十六进制数替换为固定标记

    def __hash__(self):
        """计算对象的哈希值

        Returns:
            int: 对象的哈希值
        """
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))  # 将对象的tokensWithBlanks属性转换为字符串列表，再转换为元组，计算其哈希值
    

    def __init__(self, sql, regex=False, rename=True):
        """SqlangParser类的初始化方法

        Args:
            sql (str): 输入的SQL语句
            regex (bool): 是否使用正则表达式解析字符串
            rename (bool): 是否重命名标识符
        """
        self.sql = SqlangParser.sanitizeSql(sql)  # 清理和规范化SQL语句

        self.idMap = {"COLUMN": {}, "TABLE": {}}  # 初始化列和表的标识符映射
        self.idMapInv = {}  # 初始化逆向映射
        self.idCount = {"COLUMN": 0, "TABLE": 0}  # 初始化列和表的计数器
        self.regex = regex  # 是否使用正则表达式解析字符串

        self.parseTreeSentinel = False  # 初始化解析树哨兵标志
        self.tableStack = []  # 初始化表堆栈

        self.parse = sqlparse.parse(self.sql)  # 解析SQL语句
        self.parse = [self.parse[0]]  # 只保留解析结果的第一个元素

        self.removeWhitespaces(self.parse[0])  # 移除解析树中的空白字符
        self.identifyLiterals(self.parse[0])  # 识别字面量
        self.parse[0].ptype = SUBQUERY  # 设置解析树的子查询类型
        self.identifySubQueries(self.parse[0])  # 识别子查询
        self.identifyFunctions(self.parse[0])  # 识别函数
        self.identifyTables(self.parse[0])  # 识别表

        self.parseStrings(self.parse[0])  # 解析字符串

        if rename:
            self.renameIdentifiers(self.parse[0])  # 重命名标识符

        self.tokens = SqlangParser.getTokens(self.parse)  # 获取解析树中的所有标记

    @staticmethod
    def getTokens(parse):
        """从解析树中获取所有标记

        Args:
            parse (list): 解析树列表

        Returns:
            list: 解析树中的所有标记
        """
        flatParse = []  # 初始化扁平化解析列表
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))  # 将字符串标记拆分为单词并添加到扁平化解析列表中
                else:
                    flatParse.append(str(token))  # 其他类型的标记直接添加到扁平化解析列表中
        return flatParse  # 返回扁平化解析列表

    def removeWhitespaces(self, tok):
        """移除解析树中的空白字符

        Args:
            tok (sqlparse.sql.TokenList): 解析树的根节点
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []  # 初始化临时子节点列表
            for c in tok.tokens:
                if not c.is_whitespace:  # 如果不是空白字符
                    tmpChildren.append(c)  # 添加到临时子节点列表中

            tok.tokens = tmpChildren  # 更新解析树的子节点
            for c in tok.tokens:
                self.removeWhitespaces(c)  # 递归移除子节点中的空白字符

    def identifySubQueries(self, tokenList):
        """识别解析树中的子查询

        Args:
            tokenList (sqlparse.sql.TokenList): 解析树的根节点

        Returns:
            bool: 是否为子查询
        """
        isSubQuery = False  # 初始化是否为子查询的标志

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)  # 递归识别子查询
                if subQuery and isinstance(tok, sqlparse.sql.Parenthesis):
                    tok.ttype = SUBQUERY  # 将括号类型设置为子查询类型
            elif str(tok) == "select":
                isSubQuery = True  # 如果包含SELECT关键字，则为子查询
        return isSubQuery  # 返回是否为子查询的标志

    def identifyLiterals(self, tokenList):
        """识别解析树中的字面量和标识符

        Args:
            tokenList (sqlparse.sql.TokenList): 解析树的根节点
        """
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]  # 定义空标识符的类型
        blankTokenTypes = [sqlparse.sql.Identifier]  # 定义空标识符的实例类型

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL  # 设置标记类型为内部类型
                self.identifyLiterals(tok)  # 递归识别字面量和标识符
            elif tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select":
                tok.ttype = KEYWORD  # 设置标记类型为关键字
            elif tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer:
                tok.ttype = INTEGER  # 设置标记类型为整数
            elif tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal:
                tok.ttype = HEX  # 设置标记类型为十六进制数
            elif tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float:
                tok.ttype = FLOAT  # 设置标记类型为浮点数
            elif tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol:
                tok.ttype = STRING  # 设置标记类型为字符串
            elif tok.ttype == sqlparse.tokens.Wildcard:
                tok.ttype = WILDCARD  # 设置标记类型为通配符
            elif tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0]):
                tok.ttype = COLUMN  # 设置标记类型为列

    def identifyFunctions(self, tokenList):
        """识别解析树中的函数

        Args:
            tokenList (sqlparse.sql.TokenList): 解析树的根节点
        """
        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.Function):
                self.parseTreeSentinel = True  # 如果是函数，则设置解析树哨兵标志为True
            elif isinstance(tok, sqlparse.sql.Parenthesis):
                self.parseTreeSentinel = False  # 如果是括号，则设置解析树哨兵标志为False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION  # 如果解析树哨兵标志为True，则设置标记类型为函数
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)  # 递归识别子节点中的函数


    def identifyTables(self, tokenList):
        """识别解析树中的表名

        Args:
            tokenList (sqlparse.sql.TokenList): 解析树的根节点
        """
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)  # 如果是子查询，将 False 压入表堆栈

        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]  # 获取前一个标记
            tok = tokenList.tokens[i]  # 获取当前标记

            # 识别表名的情况：如果当前标记是"."，前一个标记是列名，则前一个标记设置为表名
            if str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN:
                prevtok.ttype = TABLE

            # 识别表名的情况：如果当前标记是 "from" 关键字，将表堆栈顶部的值设置为 True
            elif str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword:
                self.tableStack[-1] = True

            # 识别表名的情况：如果当前标记是 "where"、"on"、"group"、"order" 或 "union" 关键字，将表堆栈顶部的值设置为 False
            elif (str(tok) in ["where", "on", "group", "order", "union"]) and tok.ttype == sqlparse.tokens.Keyword:
                self.tableStack[-1] = False

            # 递归处理子节点
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)

            # 如果当前标记是列名且表堆栈顶部的值为 True，则设置当前标记为表名
            elif tok.ttype == COLUMN:
                if self.tableStack[-1]:
                    tok.ttype = TABLE

        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()  # 如果是子查询，弹出表堆栈顶部的值

    def __str__(self):
        """将解析结果转换为字符串

        Returns:
            str: 解析结果的字符串表示
        """
        return ' '.join([str(tok) for tok in self.tokens])

    def parseSql(self):
        """获取解析结果的标记列表

        Returns:
            list: 解析结果的标记列表
        """
        return [str(tok) for tok in self.tokens]
#---------------------子函数2：代码的规则--------------------

#---------------------子函数3：缩略词处理--------------------
def revert_abbrev(line):
    """处理缩略词，将缩写形式转换为完整形式

    Args:
        line (str): 输入的包含缩略词的字符串

    Returns:
        str: 处理后的字符串，缩略词被转换为完整形式
    """
    # 匹配 "it", "he", "she", "that", "this", "there", "here" 后面的 "s"
    pat_is = re.compile(r"(it|he|she|that|this|there|here)\"s", re.I)
    # 匹配一般情况的 's
    pat_s1 = re.compile(r"(?<=[a-zA-Z])\"s")
    # 匹配以 "s" 结尾的 's
    pat_s2 = re.compile(r"(?<=s)\"s?")
    # 匹配否定词 not 的缩写形式
    pat_not = re.compile(r"(?<=[a-zA-Z])n\"t")
    # 匹配 would 的缩写形式
    pat_would = re.compile(r"(?<=[a-zA-Z])\"d")
    # 匹配 will 的缩写形式
    pat_will = re.compile(r"(?<=[a-zA-Z])\"ll")
    # 匹配 am 的缩写形式
    pat_am = re.compile(r"(?<=[I|i])\"m")
    # 匹配 are 的缩写形式
    pat_are = re.compile(r"(?<=[a-zA-Z])\"re")
    # 匹配 have 的缩写形式
    pat_ve = re.compile(r"(?<=[a-zA-Z])\"ve")

    # 将匹配的缩写形式替换为完整形式
    line = pat_is.sub(r"\1 is", line)  # 例如，将 "it's" 替换为 "it is"
    line = pat_s1.sub("", line)  # 删除一般情况的 's
    line = pat_s2.sub("", line)  # 删除以 "s" 结尾的 's
    line = pat_not.sub(" not", line)  # 例如，将 "can't" 替换为 "cannot"
    line = pat_would.sub(" would", line)  # 例如，将 "he'd" 替换为 "he would"
    line = pat_will.sub(" will", line)  # 例如，将 "she'll" 替换为 "she will"
    line = pat_am.sub(" am", line)  # 例如，将 "I'm" 替换为 "I am"
    line = pat_are.sub(" are", line)  # 例如，将 "they're" 替换为 "they are"
    line = pat_ve.sub(" have", line)  # 例如，将 "I've" 替换为 "I have"

    return line
#---------------------子函数3：缩略词处理--------------------

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

#---------------------子函数5：句子的去冗--------------------
def process_nl_line(line):
    """对自然语言句子进行预处理

    Args:
        line (str): 输入的自然语言句子

    Returns:
        str: 预处理后的句子
    """
    line = revert_abbrev(line)  # 处理缩略词
    line = re.sub('\t+', '\t', line)  # 将多个制表符替换为一个制表符
    line = re.sub('\n+', '\n', line)  # 将多个换行符替换为一个换行符
    line = line.replace('\n', ' ')  # 将换行符替换为空格
    line = line.replace('\t', ' ')  # 将制表符替换为空格
    line = re.sub(' +', ' ', line)  # 将多个空格替换为一个空格
    line = line.strip()  # 去除首尾空格
    line = inflection.underscore(line)  # 将骆驼命名法转换为下划线命名法

    # 去除括号里的内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 匹配括号中的内容
    line = re.sub(space, '', line)
    line = line.strip()  # 再次去除首尾空格
    return line
#---------------------子函数5：句子的去冗--------------------

#---------------------子函数6：句子的分词--------------------
def process_sent_word(line):
    """对句子进行分词和词性标注处理

    Args:
        line (str): 输入的句子

    Returns:
        list: 分词和词性处理后的单词列表
    """
    line = re.findall(r"[\w]+|[^\s\w]", line)  # 分词
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    hex_decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(hex_decimal, 'TAGINT', line)
    # 替换数字
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换特定模式字符
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")
    line = re.sub(other, 'TAGOER', line)

    cut_words = line.split(' ')  # 分词
    cut_words = [x.lower() for x in cut_words]  # 全部小写化
    word_tags = pos_tag(cut_words)  # 词性标注
    tags_dict = dict(word_tags)

    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])  # 获取词性
        if word_pos in ['a', 'v', 'n', 'r']:
            word = wnler.lemmatize(word, pos=word_pos)  # 词性还原
        word = wordnet.morphy(word) if wordnet.morphy(word) else word  # 词干提取
        word_list.append(word)
    
    return word_list
#---------------------子函数6：句子的分词--------------------

#---------------------子函数7：字符过滤（所有）--------------
def filter_all_invachar(line):
    """过滤句子中的所有非常用字符

    Args:
        line (str): 输入的句子

    Returns:
        str: 过滤后的句子
    """
    # 去除非常用符号，只保留数字、大小写字母、连字符、下划线、单引号、双引号、圆括号和换行符
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 包括\r\t也清除了
    # 替换连续的中横线为单个中横线
    line = re.sub('-+', '-', line)
    # 替换连续的下划线为单个下划线
    line = re.sub('_+', '_', line)
    # 替换竖线为空格，替换不常见的竖线为单个空格
    line = line.replace('|', ' ').replace('¦', ' ')
    return line
#---------------------子函数7：字符过滤（所有）--------------

#---------------------子函数8：字符过滤（部分）--------------
def filter_part_invachar(line):
    """过滤句子中的部分非常用字符

    Args:
        line (str): 输入的句子

    Returns:
        str: 过滤后的句子
    """
    #去除非常用符号，只保留数字、大小写字母、连字符、井号、斜杠、下划线、逗号、单引号、等号、大于号、小于号、双引号、圆括号、问号、点号、星号、方括号、尖角括号和换行符
    line= re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #包括\r\t也清除了
    # 替换连续的中横线为单个中横线
    line = re.sub('-+', '-', line)
    # 替换连续的下划线为单个下划线
    line = re.sub('_+', '_', line)
    # 替换竖线为空格，替换不常见的竖线为单个空格
    line = line.replace('|', ' ').replace('¦', ' ')
    return line
#---------------------子函数8：字符过滤（部分）--------------------

#-----------------------主函数：代码的tokens-----------------------
def sqlang_code_parse(line):
    # 使用 filter_part_invachar 过滤非常用字符
    line = filter_part_invachar(line)
    # 替换多个连续的点为单个点
    line = re.sub('\.+', '.', line)
    # 替换多个连续的制表符为单个制表符
    line = re.sub('\t+', '\t', line)
    # 替换多个连续的换行符为单个换行符
    line = re.sub('\n+', '\n', line)
    # 替换多个连续的空格为单个空格
    line = re.sub(' +', ' ', line)

    # 去除多个连续的右尖括号
    line = re.sub('>>+', '', line)
    # 替换小数为 number
    line = re.sub(r"\d+(\.\d+)+", 'number', line)

    # 去除行首和行尾的换行符，并去除多余的空格
    line = line.strip('\n').strip()
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    # 用空格连接单词
    line = ' '.join(line)

    try:
        # 使用 SqlangParser 解析 SQL 代码
        query = SqlangParser(line, regex=True)
        # 获取解析后的代码
        typedCode = query.parseSql()
        # 去除最后一个分号
        typedCode = typedCode[:-1]
        # 骆驼命名转下划线，并拆分成单词列表
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        # 去除单词列表中多余的空格，并全部转为小写
        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        token_list = [x.lower() for x in cut_tokens if x.strip() != '']  # 去除空字符串并转为小写
        return token_list
    # 如果解析失败，则返回 '-1000'
    except:
        return '-1000'
#-----------------------主函数：代码的tokens-----------------------

#-----------------------主函数：句子的tokens-----------------------
def sqlang_query_parse(line):
    """解析SQL查询语句并返回标记后的单词列表

    Args:
        line (str): 输入的SQL查询语句

    Returns:
        list: 标记后的单词列表
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list

def sqlang_context_parse(line):
    """解析SQL上下文并返回标记后的单词列表

    Args:
        line (str): 输入的SQL上下文

    Returns:
        list: 标记后的单词列表
    """
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list
#-----------------------主函数：句子的tokens-----------------------

if __name__ == '__main__':
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    # 将句子转换为代码的tokens并打印结果
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    # 将句子转换为查询语句的tokens并打印结果
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    # 将句子转换为查询语句的tokens并打印结果
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    # 将代码转换为tokens并打印结果
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    # 将代码转换为tokens并打印结果
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    # 将代码转换为tokens并打印结果
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    # 将代码转换为tokens并打印结果
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))
    # 将代码转换为tokens并打印结果

