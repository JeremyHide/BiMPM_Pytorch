# -*- coding: utf-8 -*-
"""
Multi-perspective Matching Layer.
Reference: Bilateral Multi-Perspective Matching for Natural Language Sentences.
Note: This pytorch implementation of BiMPM is based on BiMPM Keras
Sheng Liu
All rights reserved
Report bugs to ShengLiu: shengliu@nyu.edu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class MultiPerspective(nn.Module):

    def __init__(self, embedding_dim, epsilon = 1e-6, perspective = 20):
        super(MultiPerspective , self).__init__()
        self.epsilon = epsilon
        self.embedding_dim = embedding_dim
        self.perspective = perspective

        W_f1 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_f1, -0.01, 0.01)
        W_b1 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_b1, -0.01, 0.01)
        W_f2 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_f2, -0.01, 0.01)
        W_b2 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_b2, -0.01, 0.01)
        W_f3 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_f3, -0.01, 0.01)
        W_b3 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_b3, -0.01, 0.01)
        W_f4 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_f4, -0.01, 0.01)
        W_b4 = torch.Tensor(perspective, embedding_dim)
        nn.init.uniform(W_b4, -0.01, 0.01)

        self.W_f1 = nn.Parameter(W_f1)
        self.W_b1 = nn.Parameter(W_b1)
        self.W_f2 = nn.Parameter(W_f2)
        self.W_b2 = nn.Parameter(W_b2)
        self.W_f3 = nn.Parameter(W_f3)
        self.W_b3 = nn.Parameter(W_b3)
        self.W_f4 = nn.Parameter(W_f4)
        self.W_b4 = nn.Parameter(W_b4)

    def forward(self, p, q):
        p_fw = p[:, :, :self.embedding_dim]
        p_bw = p[:, :, self.embedding_dim:]
        q_fw = q[:, :, :self.embedding_dim]
        q_bw = q[:, :, self.embedding_dim:]
        #list_of_matching = []
        list_of_each_perspective = []
        matching_result_fw = self._full_matching(p_fw, q_fw, self.W_f1)
        matching_result_bw = self._full_matching(p_bw, q_bw, self.W_b1)
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching1 = torch.cat(list_of_each_perspective,1)
        
        cosine_matrix_fw = self._cosine_matrix(p_fw,q_fw)
        cosine_matrix_bw = self._cosine_matrix(p_bw,q_bw)

        matching_result_fw = self._attentive_matching(p_fw, q_fw, cosine_matrix_fw, self.W_f2)
        matching_result_bw = self._attentive_matching(p_bw, q_bw, cosine_matrix_bw, self.W_b2)
        list_of_each_perspective = []
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching2 = torch.cat(list_of_each_perspective,1)
        
        
        #list_of_matching.append(matching_result_fw)
        #list_of_matching.append(matching_result_bw)
        
        

        matching_result_fw = self._max_attentive_matching(p_fw, q_fw, cosine_matrix_fw, self.W_f3)
        matching_result_bw = self._max_attentive_matching(p_bw, q_bw, cosine_matrix_bw, self.W_b3)
        list_of_each_perspective = []
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching3 = torch.cat(list_of_each_perspective,1)
        
       

        matching_result_fw = self._max_pooling_matching(p_fw, q_fw, self.W_f4)
        matching_result_bw = self._max_pooling_matching(p_bw, q_bw, self.W_b4)
        list_of_each_perspective = []
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching4 = torch.cat(list_of_each_perspective,1)
        
        #list_of_matching.append(matching_result_fw)
        #list_of_matching.append(matching_result_bw)
        
        return torch.cat([maching1,maching2,maching3,maching4],dim = -1)





    def _cosine_similarity(self, x1, x2):
        """Compute cosine similarity.
        # Arguments:
            x1: (..., embedding_size)
            x2: (..., embedding_size)
        """
 
        cos = (x1 * x2)
        cos = cos.sum(-1)
        
        x1_norm = torch.sqrt(torch.sum(x1**2, -1).clamp(min = self.epsilon))
        x2_norm = torch.sqrt(torch.sum(x2**2, -1).clamp(min = self.epsilon))
        cos = cos / x1_norm / x2_norm
        return cos

    def _cosine_matrix(self, x1, x2):
        """Cosine similarity matrix.
        Calculate the cosine similarities between each forward (or backward)
        contextual embedding h_i_p and every forward (or backward)
        contextual embeddings of the other sentence
        # Arguments
            x1: (batch_size, x1_timesteps, embedding_size)
            x2: (batch_size, x2_timesteps, embedding_size)
        # Output shape
            (batch_size, x1_timesteps, x2_timesteps)
        """
        # expand h1 shape to (batch_size, x1_timesteps, 1, embedding_size)
        x1 = x1.unsqueeze(2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        x2 = x2.unsqueeze(1)
        # cosine matrix (batch_size, h1_timesteps, h2_timesteps)
        cos_matrix = self._cosine_similarity(x1, x2)
        return cos_matrix

    def _mean_attentive_vectors(self, x2, cosine_matrix):
        """Mean attentive vectors.
        Calculate mean attentive vector for the entire sentence by weighted
        summing all the contextual embeddings of the entire sentence
        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)
        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps, x2_timesteps, 1)
        expanded_cosine_matrix = cosine_matrix.unsqueeze(-1)
        # (batch_size, 1, x2_timesteps, embedding_size)
        x2 = x2.unsqueeze(1)
        # (batch_size, x1_timesteps, embedding_size)
        weighted_sum = (expanded_cosine_matrix * x2).sum(2)
        # (batch_size, x1_timesteps, 1)
        sum_cosine = (cosine_matrix.sum(-1) + self.epsilon).unsqueeze(-1)
        # (batch_size, x1_timesteps, embedding_size)
        attentive_vector = weighted_sum / sum_cosine
        return attentive_vector

    def _max_attentive_vectors(self, x2, cosine_matrix):
        """Max attentive vectors.
        Calculate max attentive vector for the entire sentence by picking
        the contextual embedding with the highest cosine similarity
        as the attentive vector.
        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)
        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps)
        _, max_x2_step = cosine_matrix.max(-1)

        embedding_size = x2.size()[-1]
        timesteps = max_x2_step.size()[-1]
        if timesteps is None:
            timesteps = max_x2_step.size()[-1]

        # collapse time dimension and batch dimension together
        # collapse x2 to (batch_size * x2_timestep, embedding_size)
        x2 = x2.contiguous().view(-1, embedding_size)
        # collapse max_x2_step to (batch_size * h1_timesteps)
        max_x2_step = max_x2_step.contiguous().view(-1)
        # (batch_size * x1_timesteps, embedding_size)
        max_x2 = x2[max_x2_step]
        # reshape max_x2, (batch_size, x1_timesteps, embedding_size)
        attentive_vector = max_x2.view(-1, timesteps, embedding_size)
        return attentive_vector

    def _time_distributed_multiply(self, x, w):
        """Element-wise multiply vector and weights.
        # Arguments
            x: sequence of hidden states, (batch_size, ?, embedding_size)
            w: weights of one matching strategy of one direction,
               (mp_dim, embedding_size)
        # Output shape
            (?, mp_dim, embedding_size)
        """
        # dimension of vector
        n_dim = x.dim()
        embedding_size =x.size()[-1]
        timesteps = x.size()[1]
        if timesteps is None:
            timesteps = x.size()[1]

        # collapse time dimension and batch dimension together


        x = x.contiguous().view(-1, embedding_size)
        # reshape to (?, 1, embedding_size)
        x = torch.unsqueeze(x, 1)
        # reshape weights to (1, mp_dim, embedding_size)
        w = torch.unsqueeze(w, 0)
        # element-wise multiply
        x = x * w
        # reshape to original shape
        if n_dim == 3:
            x = x.view(-1, timesteps, self.perspective, embedding_size)
            ##x.set_shape([None, None, None, embedding_size])
        elif n_dim == 2:
            x = x.view(-1, self.perspective, embedding_size)
            ##x.set_shape([None, None, embedding_size])
        return x

    def _full_matching(self, h1, h2, w):
        """Full matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h2 forward last step hidden vector, (batch_size, embedding_size)
        h2_last_state = h2[:, -1, :]
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # h2_last_state * weights, (batch_size, mp_dim, embedding_size)
        h2 = self._time_distributed_multiply(h2_last_state, w)
        # reshape to (batch_size, 1, mp_dim, embedding_size)
        h2 = h2.unsqueeze(1)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, h2)
        return matching

    def _max_pooling_matching(self, h1, h2, w):
        """Max pooling matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # h2 * weights, (batch_size, h2_timesteps, mp_dim, embedding_size)
        h2 = self._time_distributed_multiply(h2, w)
        # reshape v1 to (batch_size, h1_timesteps, 1, mp_dim, embedding_size)
        h1 = h1.unsqueeze(2)
        # reshape v1 to (batch_size, 1, h2_timesteps, mp_dim, embedding_size)
        h2 = h2.unsqueeze(1)
        # cosine similarity, (batch_size, h1_timesteps, h2_timesteps, mp_dim)
        cos = self._cosine_similarity(h1, h2)
        # (batch_size, h1_timesteps, mp_dim)
        matching = cos.max(2)[0]
        return matching

    def _attentive_matching(self, h1, h2, cosine_matrix, w):
        """Attentive matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # attentive vector (batch_size, h1_timesteps, embedding_szie)
        attentive_vec = self._mean_attentive_vectors(h2, cosine_matrix)
        # attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        attentive_vec = self._time_distributed_multiply(attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, attentive_vec)
        return matching

    def _max_attentive_matching(self, h1, h2, cosine_matrix, w):
        """Max attentive matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # max attentive vector (batch_size, h1_timesteps, embedding_szie)
        max_attentive_vec = self._max_attentive_vectors(h2, cosine_matrix)
        # max_attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        max_attentive_vec = self._time_distributed_multiply(max_attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, max_attentive_vec)
        
        return matching



