import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import sys
from models import model
import argparse
from random import shuffle
import dataloader
from dataloader import data_loader


def gradClamp(parameters, clip=5):
    for p in parameters:
        p.grad.data.clamp_(max=clip)

def train(args):

    if args.max_length < 0:
        args.max_length = 9999

    # initialize the logger
    # create logger
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

#    torch.cuda.set_device(args.gpu_id)

    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    # load train/dev/test data
    # train data
    logger.info('loading data...')

    train_loader, validation_loader, test_loader, len_chars, train_batch_num, test_batch_num = \
    data_loader(args.train_file, args.dev_file, args.test_file, \
        args.w2v_file, args.batch_size, args.max_char, args.max_sequence_length)

    logger.info('train size # sent ' + str(len(train_loader)))
    logger.info('dev size # sent ' + str(len(validation_loader)))
    logger.info('test size # sent ' + str(len(test_loader)))



    best_dev = []   # (epoch, dev_acc)

    # build the model
    bimpm = model.BiMPM(args.word_embedding_dim + args.char_rnn_dim, args.max_sequence_length, 
        len_chars + 2, args.max_char, args.char_embedding_dim, args.char_rnn_dim, args.rnn_layers, args.perspective)  
    # through to GPU
    bimpm.cuda()

    para = bimpm.parameters()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(para, lr=args.lr)
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(para, lr=args.lr)
    else:
        logger.info('No Optimizer.')
        sys.exit()

    criterion = nn.CrossEntropyLoss()
    
    logger.info('start to train...')
    

    total = 0.
    correct = 0.
    loss_data = 0.
    timer = time.time()
    
########################################3
    for epoch in range(args.epoch):
        for i, (data1, data2, lengths1, lengths2, char1, char2, char_lengths1, char_lengths2, labels) in enumerate(train_loader):
            data_batch1, data_batch2, word_length_batch1, word_length_batch2, char_batch1, char_batch2, \
            char_length_batch1, char_length_batch2, label_batch = Variable(data1.float().cuda()), Variable(data2.float().cuda()), \
            Variable(lengths1.cuda()), Variable(lengths2.cuda()), Variable(char1.cuda()), Variable(char2.cuda()), Variable(char_lengths1.cuda()), Variable(char_lengths2).cuda(), Variable(labels.cuda())

            optimizer.zero_grad()

            out = bimpm(data_batch1, data_batch2, char_batch1, char_batch2, args.hidden_size)

            loss = criterion(out,label_batch)

            loss.backward()

            gradClamp(para, clip=args.max_grad_norm)


            optimizer.step()
 

            _, predict = out.data.max(dim=1)
            total += label_batch.size(0)
            correct += torch.sum(predict == label_batch.data)
            loss_data += (loss.data[0] * args.batch_size)  # / train_lbl_batch.data.size()[0])

            if (i + 1) % args.display_interval == 0:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, time %.2fs, ' %
                            (epoch, i + 1, len(data_batch1), correct / total,
                             loss_data / label_batch.size(0), time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
            if i == len(data_batch1) - 1:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, time %.2fs, ' %
                            (epoch, i + 1, len(data_batch1), correct / total,
                             loss_data / label_batch.size(0), time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.           

        # evaluate
        if (epoch + 1) % args.dev_interval == 0:
            bimpm.eval()
            correct = 0.
            total = 0.

            
            for j, (dev_data1, dev_data2, dev_lengths1, dev_lengths2, \
                dev_char1, dev_char2, dev_char_lengths1, dev_char_lengths2, dev_labels) in enumerate(validation_loader):
                dev_data_batch1, dev_data_batch2, dev_word_length_batch1, \
                dev_word_length_batch2, dev_char_batch1, dev_char_batch2, \
                dev_char_length_batch1, dev_char_length_batch2, dev_label_batch \
                = Variable(dev_data1.float().cuda()), Variable(dev_data2.float().cuda()), Variable(dev_lengths1.cuda()), \
                Variable(dev_lengths2.cuda()), Variable(dev_char1.cuda()), Variable(dev_char2.cuda()), \
                Variable(dev_char_lengths1.cuda()), Variable(dev_char_lengths2.cuda()), Variable(dev_labels.cuda())
                

            # if dev_lbl_batch.data.size(0) == 1:
            #     # simple sample batch
            #     dev_src_batch=torch.unsqueeze(dev_src_batch, 0)
            #     dev_tgt_batch=torch.unsqueeze(dev_tgt_batch, 0)

            dev_out = bimpm(dev_data_batch1, dev_data_batch2, dev_char_batch1, dev_char_batch2, args.hidden_size)

            _, predict=dev_out.data.max(dim=1)
            total += dev_label_batch.size(0)
            correct += torch.sum(predict == dev_label_batch.data)

            dev_acc = correct / total
            logger.info('dev-acc %.3f' % (dev_acc))

            if (epoch + 1) / args.dev_interval == 1:
                model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], epoch, dev_acc)
                torch.save(bimpm.state_dict(), model_fname + 'bimpm.pt')
                best_dev.append((epoch, dev_acc, model_fname))
                logger.info('current best-dev:')
                for t in best_dev:
                    logger.info('\t%d %.3f' %(t[0], t[1]))
                logger.info('save model!') 
            else:
                if dev_acc > best_dev[-1][1]:
                    model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], epoch, dev_acc)
                    torch.save(bimpm.state_dict(), model_fname + '_bimpm.pt')
                    best_dev.append((epoch, dev_acc, model_fname))
                    logger.info('current best-dev:')
                    for t in best_dev:
                        logger.info('\t%d %.3f' %(t[0], t[1]))
                    logger.info('save model!') 

            bimpm.train()

    logger.info('training end!')
    # test
   

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_file', help = 'training data file (tsv)',
                        type=str, default = './data/train_sample.tsv')

    parser.add_argument('--dev_file', help = 'development data file (tsv)',
                        type=str, default ='./data/dev_sample.tsv')

    parser.add_argument('--test_file', help='test data file (tsv)',
                        type=str, default='./data/test_sample.tsv')

    parser.add_argument('--w2v_file', help='pretrained word vectors file (tsv)',
                        type=str, default='./data/wordvec.txt')

    parser.add_argument('--log_dir', help='log file directory',
                        type=str, default='./log')

    parser.add_argument('--log_fname', help='log file name',
                        type=str, default='log54.log')

    parser.add_argument('--max_char', help='maximal number of characters per word',
                        type=int, default=10)

    parser.add_argument('--max_sequence_length', help='maximal number of sequence',
                        type=int, default=50)

    parser.add_argument('--char_embedding_dim', help='character embedding demension',
                        type=int, default=20 )

    parser.add_argument('--perspective', help='number of perspectives',
                        type=int, default=20 )

    parser.add_argument('--word_embedding_dim', help='word embedding size',
                        type=int, default=300)

    parser.add_argument('--char_rnn_dim', help='character embedding dimension',
                        type=int, default=50)

    parser.add_argument('--rnn_layers', help='rnn layers for char embedding',
                        type=int, default=1)

    parser.add_argument('--epoch', help='training epoch',
                        type=int, default=250)

    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=32)

    parser.add_argument('--dev_interval', help='interval for development',
                        type=int, default=50)

    parser.add_argument('--optimizer', help='optimizer',
                        type=str, default='Adam')

    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.001)

    parser.add_argument('--hidden_size', help='hidden layer size',
                        type=int, default=100)

    parser.add_argument('--max_length', help='maximum length of training sentences,\
                        -1 means no length limit',
                        type=int, default=10)

    parser.add_argument('--display_interval', help='interval of display',
                        type=int, default=1000)

    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                               to have the norm equal to max_grad_norm',
                        type=float, default=5)

    parser.add_argument('--model_path', help='path of model file (not include the name suffix',
                        type=str, default='./')

    args = parser.parse_args()
    # args.max_lenght = 10   # args can be set manually like this
    train(args)

else:
    pass
