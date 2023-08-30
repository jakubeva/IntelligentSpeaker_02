import platform as plat
class ModelLanguage():  # 语音模型类
    def __init__(self, modelpath):
        self.modelpath = modelpath
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断

        self.slash = ''
        if (system_type == 'Windows'):
            self.slash = '\\'
        elif (system_type == 'Linux'):
            self.slash = '/'
        else:
            print('*[Message] Unknown System\n')
            self.slash = '/'

        if (self.slash != self.modelpath[-1]):  # 在目录路径末尾增加斜杠
            self.modelpath = self.modelpath + self.slash

        pass

    def LoadModel(self):
        self.dict_pinyin = self.GetSymbolDict('../tts_pytorch/pinyin.txt')
        model = (self.dict_pinyin)
        return model
        pass

    def TestToPinyin(self, list_words):
        length = len(list_words)
        if (length == 0):  # 传入的参数没有包含任何汉字时
            return ''

        lst_words_remain = []  # 存储剩余的汉字序列
        str_result = ''

        # 存储临时输入汉字序列
        tmp_list_words = list_words

        # while len(tmp_list_words) > 0:
            # tmp_list_result =

        # while (len(tmp_list_words) > 0):
        #     # 进行拼音转ttmp_list_syllablemp_list_syllable汉字解码，存储临时结果
        #     tmp_lst_result = self.decode(tmp_list_words, 0.0)
        #
        #     if (len(tmp_lst_result) > 0):  # 有结果，不用恐慌
        #         str_result = str_result + tmp_lst_result[0][0]
        #
        #     while (len(tmp_lst_result) == 0):  # 没结果，开始恐慌
        #         # 插入最后一个拼音
        #         lst_words_remain.insert(0, tmp_list_words[-1])
        #         # 删除最后一个拼音
        #         tmp_list_syllable = tmp_list_words[:-1]
        #         # 再次进行拼音转汉字解码
        #         tmp_lst_result = self.decode(tmp_list_syllable, 0.0)
        #
        #         if (len(tmp_lst_result) > 0):
        #             # 将得到的结果加入进来
        #             str_result = str_result + tmp_lst_result[0][0]
        #
        #     # 将剩余的结果补回来
        #     tmp_list_words = lst_words_remain
        #     lst_words_remain = []  # 清空

        return str_result


    def decode(self, list_words):
        list_pinyin = []

        num_words = len(list_words)

        # for i in num_words:


    def GetSymbolDict(self, dictfilename):
        '''
        读取拼音汉字的字典文件
        返回读取后的字典
        '''
        txt_obj = open(dictfilename, 'r', encoding='UTF-8')  # 打开文件并读入
        txt_text = txt_obj.read()
        txt_obj.close()
        txt_lines = txt_text.split('\n')  # 文本分割

        dic_symbol = {}  # 初始化符号字典
        for i in txt_lines:
            list_symbol = []  # 初始化符号列表
            if (i != ''):
                txt_l = i.split('\t')
                pinyin = txt_l[0]
                for word in txt_l[1]:
                    list_symbol.append(word)
                dic_symbol[pinyin] = list_symbol

        return dic_symbol

if __name__ == '__main__':
    ml = ModelLanguage('model_language')
    ml.LoadModel()