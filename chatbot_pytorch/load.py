import jieba
import re
import unicodedata
from chatbot_pytorch.config import *


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):  # 做分词，并且写入字典。
        # 词向量编码
        seg = jieba.lcut(sentence)
        for word in seg:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除频次小于min_count的token
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 小写、修剪和删除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def readVocs(corpus, corpus_name):
    # 文件每行读取到list lines中。
    lines = open(corpus, encoding='utf-8').read().strip().split('\n')
    # 每行用tab切分成问答两个句子，然后调用normalizeString函数进行处理。
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[s for s in l.split('|')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# 去掉低频词
def trimRareWords(voc, pairs, MIN_COUNT=0):
    # 去掉voc中频次小于3的词
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 检查问题
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查答案
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        # 如果问题和答案都只包含高频词，我们才保留这个句对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


# 结束语句EOS_token
def filterPair(p):
    # 词向量编码
    seg_list_q = jieba.lcut(p[0])
    seg_list_a = jieba.lcut(p[1])
    return len(seg_list_q) < MAX_LENGTH and len(seg_list_a) < MAX_LENGTH
    # return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# 过滤太长的句对
def filterPairs(pairs):
    A = [pair for pair in pairs if filterPair(pair)]
    return [pair for pair in pairs if filterPair(pair)]


# 使用上面的函数进行处理，返回Voc对象和句对的list
def prepareData(corpus, corpus_name):
    voc, pairs = readVocs(corpus, corpus_name)
    pairs = filterPairs(pairs)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    # 在原有的基础上再添加trimRareWords，过滤低频词
    # pairs = trimRareWords(voc, pairs)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs


# 加载数据
def loadPrepareData(corpus):
    # corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
        print('已有处理好的数据集！')
    except FileNotFoundError:
        print('没有处理好的数据集！正在处理生成中...')
        voc, pairs = prepareData(corpus, corpus_name)
        print('数据集处理完毕，数据tar包生成成功！')
    return voc, pairs
