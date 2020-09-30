import torch
import torch.autograd as autograd
import torch.nn as nn
START_TAG = -2
STOP_TAG = -1

def log_sum_exp(vec, m):
    _, index = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, index.view(-1, 1, m)).view(-1, 1, m)
    return max_score.view(-1, 1, m) + torch.log(torch.sum(torch.exp(vec-max_score.expand_as(vec)), 1)).view(-1 ,1, m)

class CRF(nn.module):
    def __init__(self, tagset_size, gpu):
        '''
        :param tagset_size: 输入数据的size
        :param gpu:gpu的配置
        这个函数用作初始化，其中的transitions是一个过度矩阵，作用好像是给后面的一个量赋值...
        '''
        super(CRF, self).__init__()
        print("Preparing CRF layer...")

        self.tag_size = tagset_size
        self.gpu = gpu

        # We add 2 here, because of START_TAG and STOP_TAG
        # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        temp = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        if self.gpu:
            temp = temp.cuda()
        self.transitions = nn.Parameter(temp)  # (t+2,t+2)


    def get_iter(self, feats, mask):
        """
        :return: 获取供迭代器，提供转化后的partition， scores， mask
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        # 转化mask的维度方便计算
        mask = mask.transpose(1, 0).contiguous()
        counts = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(counts, 1, tag_size).expand(counts, tag_size, tag_size) # (i,t+2,t+2) 第2维t+2的每一个是一样的
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(counts, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # 构建迭代器
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()  # (batch_size,from_target_size,to_target_size) inivalues是每个句子的第一个字
        # 只需要从start_tag开始
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size,
                                                            1)  # batch_size * to_target_size (b,t,1)
        return seq_iter, partition, scores, mask, feats

    def _calculate_PZ(self, feats, mask):
        """
        :param feats: 输入的数据，其格式为（batch, seq_len, tag_size+2)
        :param mask: 评价的tag, 其格式为（batch, seq_len)
        :return: 序列的score的一部分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.tagset_size + 2)

        seq_iter, partition, scores, mask, feats = self.get_iter(self, feats, mask)

        for idx, cur_values in seq_iter:
            '''
            上一个 to_target 是现在的 from_target
            partition: 之前计算的log(exp(from_target)), (batch_size * from_target)
            cur_values: (batch_size , from_target , to_target)
            '''

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size) # 聚合之前计算的和
            cur_log_sum = log_sum_exp(cur_values, tag_size)  # (batch_size,tag_size)

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            # 有效更新tag部分，只保留mask value=1的值
            masked_cur_log_sum = cur_log_sum.masked_select(mask_idx)
            # 按照原文所说，这里是为了避免报错
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            partition.masked_scatter_(mask_idx, masked_cur_log_sum)
            # 在到达结束状态之前，为所有partition添加转换分数（log_sum_exp），然后选择STOP_TAG中的值
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size,
                                                                         tag_size) + \
                     partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

        cur_log_sum = log_sum_exp(cur_values, tag_size)  # (batch_size,hidden_dim)
        final_log_sum = cur_log_sum[:, STOP_TAG]  # (batch_size) log_sum储存在最后一维
        return final_log_sum.sum(), scores  # scores: (seq_len, batch, tag_size, tag_size)

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) 解码的序列
                path_score: (batch, 1) 各序列对应得分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.tagset_size + 2)
        seq_iter, partition, scores, mask, feats = self.get_iter(self, feats, mask)

        # 计算每个句子的长度
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # 动态规划记录最优点的路径
        back_points = list()
        log_sum_history = list()
        # 将mask反转，为了不出现bug这样修改
        mask = (1 - mask.long()).byte()
        log_sum_history.append(partition)  # (seqlen,batch_size,tag_size,1)
        log_sum = partition.contiguous()


        for idx, cur_values in seq_iter:
            # 上一个 to_target 是现在的 from_target
            # log_sum: 之前计算的log(exp(from_target)), (batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + log_sum.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            log_sum, cur_bp = torch.max(cur_values, dim=1)
            log_sum_history.append(log_sum.unsqueeze(2))

            # cur_bp: (batch_size, tag_size) 当前标记中的最大源分数位置
            # 将被填充标签设置为0，这将在后期处理中过滤
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        # 将分数添加到STOP_TAG
        log_sum_history = torch.cat(log_sum_history, dim=0).view(seq_len, batch_size, -1).transpose(1,
                                                                                    0).contiguous()  # (batch_size, seq_len, tag_size)
        # 获取每个句子的最后一个位置，并使用gather（）选择最后一个partitions
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(log_sum_history, 1, last_position).view(batch_size, tag_size, 1)

        # 计算从最后一个partition到结束状态的分数（然后从中选择STOP_标记）
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size,
                                                                                                    tag_size)\
            .expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)  # (batch_size,tag_size)

        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()

        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        # select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]  # (batch_size)
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()  # (batch_size,sq_len,tag_size)
        # 将结束index（expand to tag_size）移动到back_points的相应位置以替换0值

        back_points.scatter_(1, last_position, insert_last)  ##batch_size,sq_len,tag_size)
        back_points = back_points.transpose(1, 0).contiguous()  # (seq_len, batch_size, tag_size)
        # 从末尾解码，填充的位置 index 为0
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1,
                                   pointer.contiguous().view(batch_size, 1))  # pointer:(batch_size,1)
            decode_idx[idx] = pointer.squeeze(1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)  # (batch_size, sent_len)
        return path_score, decode_idx  #

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores:  (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags:  (batch, seq_len)
            output:
                score: gold sorce 的和，在一整个batch内
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)

        # 将标记值转换为另外一种格式，将记录的标签转换为index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                # start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        # transition for label to STOP_TAG
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        # 一个batch的长度,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # 标记最后一个词的label的id
        end_ids = torch.gather(tags, 1, length_mask - 1)

        # 标记转化分数，记录在STOP_TAG维度上
        end_energy = torch.gather(end_transition, 1, end_ids)

        # 将 tag 转化为 (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        # 需要转换tag id
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len,
                                                                                         batch_size)  # seq_len * bat_size
        # mask to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # loss函数
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats,
                                                   mask)  # forward_score: long, scores: (seq_len, batch, tag_size, tag_size)
        gold_score = self._score_sentence(scores, mask, tags)

        return forward_score - gold_score
