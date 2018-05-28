import time
import argparse
import torch
import msgpack
from SQuAD.model.model import DocReaderModel
from SQuAD.model.utils import str2bool
from prepro import annotate, to_id, init
from SQuAD.train import BatchGen


def init_model(args):
    if args.cuda:
        checkpoint = torch.load(args.model_file)
    else:
        checkpoint = torch.load(args.model_file, map_location=lambda storage, loc: storage)

    state_dict = checkpoint['state_dict']
    opt = checkpoint['config']

    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    embedding = torch.Tensor(meta['embedding'])

    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    opt['cuda'] = args.cuda
    opt['qanet_tail'] = opt.get('qanet_tail', False)

    BatchGen.pos_size = opt['pos_size']
    BatchGen.ner_size = opt['ner_size']

    model = DocReaderModel(opt, embedding, state_dict)
    if args.cuda:
        model.cuda()

    w2id = {w: i for i, w in enumerate(meta['vocab'])}
    tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
    ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}
    init()

    return meta, w2id, tag2id, ent2id, model
