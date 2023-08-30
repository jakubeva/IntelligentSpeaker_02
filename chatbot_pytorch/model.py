import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SOS_token, device

# Encoder模型，把变长的输入序列编码成一个固定长度的context向量
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0):
        """
        初始化
        :param input_size: 输入大小
        :param hidden_size: 隐状态大小
        :param embedding: 映射层
        :param n_layers:
        :param dropout:
        """
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 把词的ID通过Embedding层变成向量
        embedded = self.embedding(input_seq)
        # 把Embedding后的数据进行pack
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # 传入GRU进行Forward计算
        outputs, hidden = self.gru(packed, hidden)  # output: (seq_len, batch, hidden*n_dir)
        # Unpack计算结果
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 把双向GRU的结果向量加起来
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # 返回(所有时刻的)输出和最后时刻的隐状态
        return outputs, hidden


# Luong attention layer 注意力机制
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions，转置
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)，把score变成概率
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# Decoder将固定长度的向量变成可变长度的目标的信号序列
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU，通过单向GRU向前传播
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights from the current GRU output，使用新的隐状态计算注意力权重
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector，用注意力权重得到context向量
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        # context向量和GRU的输出拼接起来，然后再进过一个全连接网络，使得输出大小仍然是hidden_size
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)

        # 激活函数
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6，使用一个投影矩阵把输出从hidden_size变成词典大小，然后用softmax变成概率
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        # Return output and final hidden state，返回输出和新的隐状态
        return output, hidden


# 贪婪搜索解码器
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # encoder编码器的最终隐藏层作为decoder解码器的第一个隐藏输入
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # SOS_token初始化解码器输入
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # 初始化张量
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 一次迭代解码一个词标记
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 获取最可能的词标记及其softmax概率分数
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 记录token和score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 准备当前token作为下一个解码器输入（需要添加维度）
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


# 集束搜索解码器
class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, beam_width):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beam_width = beam_width

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # 初始化解码器的输入为起始标记
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        # 初始化beam宽度为1的候选项
        beam = [(decoder_input, decoder_hidden, 0)]

        for _ in range(max_length):
            candidates = []
            for decoder_input, decoder_hidden, score in beam:
                # 对于每个候选项，将其输入到解码器中，得到解码器的输出和更新后的隐藏状态
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                # 使用torch.topk函数选择得分最高的beam_width个标记作为下一步的候选项
                decoder_scores, decoder_input = torch.topk(decoder_output, self.beam_width)

                # 将每个候选项的标记、隐藏状态和累积得分保存在candidates列表中
                for i in range(self.beam_width):
                    candidate = (decoder_input[:, i].unsqueeze(1), decoder_hidden, score + decoder_scores[:, i])
                    candidates.append(candidate)

            # 根据分数选择顶级候选项
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:self.beam_width]

        # 从最终的beam中提取输出标记和分数
        all_tokens = torch.cat([candidate[0] for candidate in beam], dim=0)
        all_scores = torch.cat([candidate[2] for candidate in beam], dim=0)

        return all_tokens, all_scores