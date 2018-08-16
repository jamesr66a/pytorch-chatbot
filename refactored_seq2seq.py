import torch
import numpy as np

import torch.nn.functional as F

class GRUEncoder(torch.nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0.0):
        super(GRUEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input, seq_lens, hidden=None):
        input = self.embedding(input)
        input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_lens)
        outputs, hidden = self.gru(input, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

T, B, C = 20, 5, 512
n_words = 10000
embedding = torch.nn.Embedding(n_words, C)
pytorch_encoder = GRUEncoder(512, embedding)

tracing_inputs = torch.LongTensor(T, B).random_(0, n_words)
tracing_seq_len = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=T, size=(B,))))
print(tracing_inputs.size(), tracing_seq_len.size())
traced_encoder = torch.jit.trace(tracing_inputs, tracing_seq_len)(pytorch_encoder)

print(traced_encoder.graph)

test_inputs = torch.LongTensor(T, B).random_(0, n_words)
test_seq_len = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=T, size=(B,))))

np.testing.assert_allclose(pytorch_encoder(test_inputs, test_seq_len)[0].detach().numpy(),
                           traced_encoder(test_inputs, test_seq_len)[0].detach().numpy())


class Attention(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        return F.softmax(attn_energies, dim=1).unsqueeze(2)

traced_encoder_out = traced_encoder(test_inputs, test_seq_len)[0]

for attn_type in ['general', 'concat', 'dot']:
    pytorch_attn = Attention(attn_type, C)
    attn_input_hidden = torch.randn(1, B, C)
    traced_attn = torch.jit.trace(attn_input_hidden, traced_encoder_out)(pytorch_attn)

    print(traced_attn.graph)
    attn_input_hidden = torch.randn(1, B, C)
    np.testing.assert_allclose(pytorch_attn(attn_input_hidden, traced_encoder_out).detach().numpy(),
                               traced_attn(attn_input_hidden, traced_encoder_out).detach().numpy())


class AttentionDecoderGRU(torch.nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttentionDecoderGRU, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)

        self.attn = Attention(attn_model, hidden_size)

    def forward(self, input_one_step, last_hidden, encoder_outputs):
        tokens_embedded = self.embedding(input_one_step)
        tokens_embedded = self.embedding_dropout(tokens_embedded)

        rnn_output, hidden = self.gru(tokens_embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.transpose(0, 1).transpose(1, 2).bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1).unsqueeze(0)
        return output, hidden

decoder_embedding = torch.nn.Embedding(n_words, C)

for attn_type in ['general', 'concat', 'dot']:
    print('*****', attn_type)
    pytorch_decoder = AttentionDecoderGRU(attn_type, decoder_embedding, C, n_words)
    pytorch_decoder.eval() # Disable dropout for testing
    input_one_step = torch.LongTensor(1, B).random_(0, n_words)
    fake_decoder_hidden = torch.randn(1, B, C)
    traced_decoder = torch.jit.trace(input_one_step, fake_decoder_hidden, traced_encoder_out)(pytorch_decoder)
    print(traced_decoder.graph)

    input_t_pytorch = input_one_step.clone()
    hidden_t_pytorch = fake_decoder_hidden.clone()
    input_t_traced = input_one_step.clone()
    hidden_t_traced = fake_decoder_hidden.clone()


    n_timesteps = 20
    for i in range(n_timesteps):
        print(input_t_pytorch[0, :], input_t_traced[0, :])
        scores_t_pytorch, hidden_t_pytorch = pytorch_decoder(input_t_pytorch, hidden_t_pytorch, traced_encoder_out)
        scores_t_traced, hidden_t_traced = traced_decoder(input_t_traced, hidden_t_traced, traced_encoder_out)
        np.testing.assert_allclose(input_t_pytorch.detach().numpy(), input_t_traced.detach().numpy())
        np.testing.assert_allclose(hidden_t_pytorch.detach().numpy(), hidden_t_traced.detach().numpy())

        input_t_pytorch = torch.argmax(scores_t_pytorch, dim=2)
        input_t_traced = torch.argmax(scores_t_traced, dim=2)



class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, T, B, C, n_words):
        super(GreedySearchDecoder, self).__init__()
        test_inputs = torch.LongTensor(T, B).random_(0, n_words)
        test_seq_len = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=T, size=(B,))))
        self.encoder_embed = torch.nn.Embedding(n_words, C)
        self.encoder = torch.jit.trace(test_inputs, test_seq_len)(GRUEncoder(C, self.encoder_embed))
        test_outputs, test_hidden = self.encoder(test_inputs, test_seq_len)
        input_one_step = torch.LongTensor(1, B).random_(0, n_words)
        fake_decoder_hidden = torch.randn(1, B, C)
        self.decoder = torch.jit.trace(input_one_step, fake_decoder_hidden, traced_encoder_out)(AttentionDecoderGRU(attn_type, decoder_embedding, C, n_words))
        self.C = C

    __constants__ = ['C']

    @torch.jit.script_method
    def forward(self, input_tokens, input_seq_lens, n_timesteps, input_t, all_tokens, all_scores):
        # type: (Tensor, Tensor, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        enc_outs, enc_hiddens = self.encoder(input_tokens, input_seq_lens)
        # TODO: fix the lack of type casts
        # input_t = torch.zeros([1, input_tokens.size(1)])
        hidden_t = torch.zeros([1, input_tokens.size(1), self.C])

        # all_tokens = torch.zeros([0, input_tokens.size(1)])
        for i in range(n_timesteps):
            prob_t, hidden_t = self.decoder(input_t, hidden_t, enc_outs)
            scores_t, input_t = torch.max(prob_t, dim=2)

            all_tokens = torch.cat((all_tokens, input_t), dim=0)
            all_scores = torch.cat((all_scores, scores_t), dim=0)

        return all_tokens, all_scores

search = GreedySearchDecoder(T=20, B=5, C=512, n_words=10000)
print(search.graph)
test_inputs = torch.LongTensor(20, 5).random_(0, 10000)
test_seq_len = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=20, size=(B,))))

input_t = torch.zeros([1, test_inputs.size(1)]).long()
all_tokens = torch.zeros([0, test_inputs.size(1)]).long()
all_scores = torch.zeros([0, test_inputs.size(1)])

print(search(test_inputs, test_seq_len, 20, input_t, all_tokens, all_scores))

import math

class MILSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MILSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter("bias", None)
        self.alpha = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_h = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_i = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hx, cx):
        # get prev_t, cell_t from states
        Wx = F.linear(x, self.weight_ih)
        Uz = F.linear(hx, self.weight_hh)

        # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
        gates = self.alpha * Wx * Uz + self.beta_i * Wx + self.beta_h * Uz + self.bias

        # Same as LSTMCell after this point
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy


class MILSTMEncoder(torch.jit.ScriptModule):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0.0):
        super(MILSTMEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.initial_hidden = torch.nn.Parameter(torch.randn(hidden_size))
        self.initial_cell = torch.nn.Parameter(torch.randn(hidden_size))

        self.milstm = torch.jit.trace(torch.rand(1, 5, hidden_size), self.initial_hidden.repeat(1, 5, 1), self.initial_cell.repeat(1, 5, 1))(MILSTMCell(hidden_size, hidden_size))

    @torch.jit.script_method
    def forward(self, input, seq_lens):
        input = self.embedding(input)
        hidden = self.initial_hidden.unsqueeze(0).repeat([1, input.size(1), 1])
        cell = self.initial_cell.unsqueeze(0).repeat([1, input.size(1), 1])
        outputs = torch.zeros([0, input.size(1), input.size(2)])
        for i in range(input.size(0)):
            hidden, cell = self.milstm(input[i], hidden, cell)
            outputs = torch.cat((outputs, hidden), dim=0)

        return outputs, hidden

milstm = MILSTMEncoder(C, embedding)

inputs = torch.LongTensor(20, 5).random_(0, 10000)
seq_lens = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=20, size=(5,))))
outs = milstm(inputs, seq_lens)
print(milstm.graph)


class AttentionDecoderMILSTM(torch.nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttentionDecoderMILSTM, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.milstm = torch.jit.trace(torch.rand(1, 5, hidden_size), torch.rand(1, 5, hidden_size), torch.rand(1, 5, hidden_size))(MILSTMCell(hidden_size, hidden_size))
        self.concat = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)

        self.attn = Attention(attn_model, hidden_size)

    def forward(self, input_one_step, last_hidden, last_cell, encoder_outputs):
        tokens_embedded = self.embedding(input_one_step)
        tokens_embedded = self.embedding_dropout(tokens_embedded)

        rnn_output, cell = self.milstm(tokens_embedded, last_hidden, last_cell)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.transpose(0, 1).transpose(1, 2).bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1).unsqueeze(0)
        return output, rnn_output, cell

class MILSTMGreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, T, B, C, n_words):
        super(MILSTMGreedySearchDecoder, self).__init__()
        test_inputs = torch.LongTensor(T, B).random_(0, n_words)
        test_seq_len = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=T, size=(B,))))
        self.encoder_embed = torch.nn.Embedding(n_words, C)
        self.encoder = MILSTMEncoder(C, self.encoder_embed)
        test_outputs, test_hidden = self.encoder(test_inputs, test_seq_len)
        input_one_step = torch.LongTensor(1, B).random_(0, n_words)
        fake_decoder_hidden = torch.randn(1, B, C)
        self.decoder = torch.jit.trace(input_one_step, fake_decoder_hidden, fake_decoder_hidden, traced_encoder_out)(AttentionDecoderMILSTM(attn_type, decoder_embedding, C, n_words))
        self.C = C

    __constants__ = ['C']

    @torch.jit.script_method
    def forward(self, input_tokens, input_seq_lens, n_timesteps, input_t, all_tokens, all_scores):
        # type: (Tensor, Tensor, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        enc_outs, enc_hiddens = self.encoder(input_tokens, input_seq_lens)
        # TODO: fix the lack of type casts
        # input_t = torch.zeros([1, input_tokens.size(1)])
        hidden_t = torch.zeros([1, input_tokens.size(1), self.C])
        cell_t = torch.zeros([1, input_tokens.size(1), self.C])

        # all_tokens = torch.zeros([0, input_tokens.size(1)])
        for i in range(n_timesteps):
            prob_t, hidden_t, cell_t = self.decoder(input_t, hidden_t, cell_t, enc_outs)
            scores_t, input_t = torch.max(prob_t, dim=2)

            all_tokens = torch.cat((all_tokens, input_t), dim=0)
            all_scores = torch.cat((all_scores, scores_t), dim=0)

        return all_tokens, all_scores


milstm_search = MILSTMGreedySearchDecoder(T=20, B=5, C=512, n_words=10000)
print(milstm_search.graph)
test_inputs = torch.LongTensor(20, 5).random_(0, 10000)
test_seq_len = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=20, size=(B,))))

input_t = torch.zeros([test_inputs.size(1)]).long()
all_tokens = torch.zeros([0, test_inputs.size(1)]).long()
all_scores = torch.zeros([0, test_inputs.size(1)])

print(milstm_search(test_inputs, test_seq_len, 20, input_t, all_tokens, all_scores))
