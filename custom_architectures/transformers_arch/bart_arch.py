# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 BART model. """

import tensorflow as tf

from hparams.hparams import HParams
from custom_layers import FasterEmbedding
from custom_architectures.transformers_arch.transformer_arch import *

__base_bart_config = {
    'vocab_size'    : -1,
    'embedding_dim' : -1,
    'max_input_length'  : 1024,
    'scale_embedding'   : False,
    'return_attention'  : True,
    'epsilon'   : 1e-5
}

HParamsBartEncoder  = HParamsTransformerEncoder(** __base_bart_config)

HParamsBartDecoder  = HParamsTransformerDecoder(** __base_bart_config)

HParamsBart         = HParamsTransformer(
    ** HParamsTransformerEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsTransformerDecoder.get_config(add_prefix = 'decoder'),
    ** __base_bart_config
)

class BartEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, token_embedding, name = None, ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsBartEncoder.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        self.vocab_size     = vocab_size
        self.embedding_dim  = embedding_dim
        
        
        self.embedding_factor = tf.math.sqrt(float(embedding_dim)) if self.hparams.scale_embedding else 1.
        
        self.token_embedding_layer = token_embedding
        self.pos_embedding_layer   = FasterEmbedding(
            self.hparams.max_input_length + 2, self.embedding_dim, name = "pos_embeddings"
        )
        self.encoder    = TransformerEncoder(** self.hparams, name = "encoder")
        
        self.norm       = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)

    def _build(self):
        batch_size, seq_len = 2, 32
        text = tf.ones([batch_size, seq_len], dtype = tf.int32)
        text_length = tf.fill([batch_size, 1], seq_len)
        
        self([text, text_length], training = False)
        
    def freeze(self, trainable = False):
        self.token_embedding_layer.trainable    = trainable
        self.pos_embedding_layer.trainable      = trainable
        self.encoder.trainable  = trainable 
        self.norm.trainable     = trainable 

    def call(self, inputs, mask = None, training = False, return_attention = None):
        if return_attention is None: return_attention = self.hparams.return_attention
        text, text_lengths = inputs
        batch_size, seq_len = tf.shape(text)[0], tf.shape(text)[1]
        
        token_embedded = self.token_embedding_layer(text) * self.embedding_factor
        pos_embedded   = self.pos_embedding_layer(tf.range(seq_len) + 2)
        
        embedded = self.norm(token_embedded + pos_embedded)
        embedded = self.dropout(embedded, training = training)
        
        if mask is None:
            mask    = tf.sequence_mask(
                text_lengths, maxlen = seq_len, dtype = tf.float32
            )
        
        mask = 1. - tf.reshape(tf.cast(mask, tf.float32), [batch_size, 1, 1, seq_len])

        encoder_outputs, attn_weights = self.encoder(
            embedded, mask = mask, training = training, return_attention = True
        )
        
        return encoder_outputs, attn_weights if return_attention else encoder_outputs

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

    
class BartDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, token_embedding, name = None, ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsBartDecoder.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        self.vocab_size     = vocab_size
        self.embedding_dim  = embedding_dim
        
        self.embedding_factor = tf.math.sqrt(float(embedding_dim)) if self.hparams.scale_embedding else 1.
        
        self.token_embedding_layer = token_embedding
        self.pos_embedding_layer   = FasterEmbedding(
            self.hparams.max_input_length + 2, self.embedding_dim, name = "pos_embeddings"
        )
        self.decoder    = TransformerDecoder(** self.hparams, name = "decoder")
        
        self.norm       = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)

    def _build(self):
        batch_size, in_seq_len, out_seq_len = 2, 8, 16
        
        encoder_out = tf.random.normal((batch_size, in_seq_len, self.embedding_dim))
        text    = tf.ones([batch_size, out_seq_len], dtype = tf.int32)
        
        self([encoder_out, text], training = False)
        
    def freeze(self, trainable = False):
        self.token_embedding_layer.trainable    = trainable
        self.pos_embedding_layer.trainable      = trainable
        self.decoder.trainable  = trainable 
        self.norm.trainable     = trainable 
        

    def call(self,
             inputs,
             padding_mask   = None,
             look_ahead_mask    = None,
             training       = False,
             return_attention   = None
            ):
        if return_attention is None: return_attention = self.hparams.return_attention
        encoder_out, text = inputs
        
        batch_size  = tf.shape(text)[0]
        seq_len     = tf.shape(text)[1]
        
        token_embedded = self.token_embedding_layer(text) * self.embedding_factor
        pos_embedded   = self.pos_embedding_layer(tf.range(seq_len) + 2)
        
        embedded = self.norm(token_embedded + pos_embedded)
        embedded = self.dropout(embedded, training = training)

        decoder_outputs, attn_weights = self.decoder(
            [encoder_out, embedded], training = training, return_attention = True, look_ahead_mask = look_ahead_mask
        )
        
        return decoder_outputs, attn_weights if return_attention else decoder_outputs

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class Bart(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, max_input_length, name = None, ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsTransformer.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size      = vocab_size,
            embedding_dim   = embedding_dim,
            max_input_length    = max_input_length,
            encoder_embedding_dim   = embedding_dim,
            decoder_embedding_dim   = embedding_dim
        )        
        self.vocab_size     = vocab_size
        self.embedding_dim  = embedding_dim
        
        self.shared_embedding = FasterEmbedding(vocab_size, embedding_dim, name = "token_embedding")
        
        self.encoder    = BartEncoder(
            vocab_size, max_input_length = max_input_length, token_embedding = self.shared_embedding, 
            name = "encoder", ** self.hparams.get_config(prefix = 'encoder')
        )
        self.decoder    = BartDecoder(
            vocab_size, max_input_length = max_input_length, token_embedding = self.shared_embedding, 
            name = "decoder", ** self.hparams.get_config(prefix = 'decoder')
        )
        
        self.final_bias = self.add_weight(
            shape = [1, vocab_size], name = "final_bias", trainable = False, initializer = "zeros"
        )

    def _build(self):
        batch_size, in_seq_len, out_seq_len = 2, 16, 32
        text_in = tf.ones([batch_size, in_seq_len], dtype = tf.int32)
        text_in_length = tf.fill([batch_size, 1], in_seq_len)
        text_out = tf.ones([batch_size, out_seq_len], dtype = tf.int32)
        text_out_length = tf.fill([batch_size, 1], out_seq_len)
        
        self([text_in, text_in_length, text_out, text_out_length], training = False)
        
    def freeze(self, trainable = False):
        self.token_embedding.trainable = trainable 
        self.encoder.trainable = trainable 
        self.decoder.trainable = trainable

    def call(self, inputs, mask = None, look_ahead_mask = None, training = False, return_attention = True):
        """
            Perform BERT inference
            
            Arguments :
                - inputs    : list of length 2 or 3
                    text        : tokens with shape [batch_size, seq_len]
                    text_length : text length with shape [batch_size, 1]
                    token_type  : (Optional) type of token with shape [batch_size, seq_len]
                - mask      : padding mask (if not provided, built based on `text_length`)
                - training  : bool, whether it is training phase or not 
                - return_attention  : whether to return attention_scores for the TransformerEncoder
            Return : (output, pooled_output, attn_weights) if return_attention else output
                - output    : Encoder output with shape [batch_size, seq_len, embedding_dim]
                - attn_weights  : dict of Encoder attention weights
                    Each value corresponds to attention of a given layer with shape [batch_size, num_heads, seq_len, seq_len]
                
        """
        text_in, text_in_length, text_out, text_out_length = inputs
        
        encoder_out, encoder_attn_weights = self.encoder(
            [text_in, text_in_length], training = training, return_attention = True
        )
        decoder_outputs, decoder_attn_weights = self.decoder(
            [encoder_out, text_out], training = training, return_attention = True, look_ahead_mask = look_ahead_mask
        )
        
        batch_size, out_seq_len = tf.shape(decoder_outputs)[0], tf.shape(decoder_outputs)[1]
        
        output = tf.reshape(decoder_outputs, [-1, self.embedding_dim])
        output = tf.matmul(output, self.shared_embedding.embeddings, transpose_b = True)
        output = tf.reshape(output, [batch_size, out_seq_len, self.vocab_size])
        
        output = output + self.final_bias
        
        return output, decoder_attn_weights if return_attention else output

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

    @classmethod
    def from_pretrained(cls, pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation', ** kwargs):
        from models.weights_converter import partial_transfer_learning
        
        with tf.device('cpu') as d:
            pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBart(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,

            encoder_num_layers  = pretrained.config.encoder_layers,
            encoder_ffn_dim     = pretrained.config.encoder_ffn_dim,
            encoder_ffn_activation  = pretrained.config.activation_function,
            encoder_mha_num_heads   = pretrained.config.encoder_attention_heads,
            encoder_mha_epsilon     = 1e-5,
            encoder_epsilon     = 1e-5,

            decoder_num_layers  = pretrained.config.decoder_layers,
            decoder_ffn_dim     = pretrained.config.decoder_ffn_dim,
            decoder_ffn_activation  = pretrained.config.activation_function,
            decoder_mha_num_heads   = pretrained.config.decoder_attention_heads,
            decoder_mha_epsilon     = 1e-5,
            decoder_enc_mha_num_heads   = pretrained.config.decoder_attention_heads,
            decoder_enc_mha_epsilon     = 1e-5,
            decoder_epsilon     = 1e-5
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        offset, n_enc_layer_weights = 2, 16
        
        weights = pretrained.get_weights()
        # Invert `key` and `value` weights for each MHA layer
        for i in range(pretrained.config.encoder_layers):
            weights[i * n_enc_layer_weights + offset], weights[i * n_enc_layer_weights + offset + 2] = (
                weights[i * n_enc_layer_weights + offset + 2], weights[i * n_enc_layer_weights + offset]
            )
            weights[i * n_enc_layer_weights + offset + 1], weights[i * n_enc_layer_weights + offset + 3] = (
                weights[i * n_enc_layer_weights + offset + 3], weights[i * n_enc_layer_weights + offset + 1]
            )


        offset = n_enc_layer_weights * pretrained.config.encoder_layers + offset + 3
        n_mha_weights, n_dec_layer_weights = 10, 26
        for i in range(pretrained.config.decoder_layers):
            weights[i * n_dec_layer_weights + offset], weights[i * n_dec_layer_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + offset + 2], weights[i * n_dec_layer_weights + offset]
            )
            weights[i * n_dec_layer_weights + offset + 1], weights[i * n_dec_layer_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + offset + 3], weights[i * n_dec_layer_weights + offset + 1]
            )
            
            weights[i * n_dec_layer_weights + n_mha_weights + offset], weights[i * n_dec_layer_weights + n_mha_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 2],
                weights[i * n_dec_layer_weights + n_mha_weights + offset]
            )
            weights[i * n_dec_layer_weights + n_mha_weights + offset + 1], weights[i * n_dec_layer_weights + n_mha_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 3],
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 1]
            )
        
        partial_transfer_learning(instance, weights)
        
        return instance

def transformers_bart(name = 'facebook/bart-large', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.TFBartForConditionalGeneration.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

custom_functions    = {
    'transformers_bart' : transformers_bart,
    
    'BartEncoder'   : BartEncoder,
    'BartDecoder'   : BartDecoder,
    'Bart'          : Bart
}

custom_objects  = {
    'TransformerEncoder'    : TransformerEncoder,
    
    'BartEncoder'   : BartEncoder,
    'BartDecoder'   : BartDecoder,
    'Bart'          : Bart
}

