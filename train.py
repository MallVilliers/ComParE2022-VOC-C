import argparse
import time
import warnings

import pandas as pd
from torch.utils.data import DataLoader

from data_loader import TrainLoader, load_feature
from model_trainer import ModelTrainer
from utils import init_args, cm_analysis

warnings.simplefilter("ignore")


def main(args):
    """
    模型训练主函数

    Parameter：
        args:参数配置项

    """

    # 加载数据
    if args.feature_type:
        feat_df = load_feature(args.feature_type, args.feature_path)
    else:
        feat_df = pd.DataFrame()
    dataset = TrainLoader(feat_df, **vars(args))
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.n_cpu,
                             drop_last=False)

    # 初始化trainer
    trainer = ModelTrainer(**vars(args))

    # 记录结果
    uars = [0]
    score_file = open(args.score_save_path, "a+")
    score_file.write(f'Model: {args.model} \n'
                     f'Feature Type: {args.feature_type} \n'
                     f'Spectrogram : {args.mel}, n_mel: {args.n_mel} \n'
                     + '-' * 20 + '\n')

    # 模型训练
    for epoch in range(1, args.epoch + 1):
        print("-" * 10, f'第{epoch}轮训练开始', "-" * 10)

        # 训练
        lr, loss = trainer.train_network(epoch, data_loader, args.mel, args.n_mel)
        # 测试
        if epoch % args.test_step == 0:
            trainer.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
            uar, true_list, pred_list = trainer.eval_network(args.devel_path, args.wav_path, args.feature_type, feat_df,
                                                             args.mel, args.n_mel)
            if uar > max(uars):
                cm_analysis(true_list, pred_list, args.save_path + '/cm', sorted(set(true_list)))
            uars.append(uar)

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, LOSS %f, UAR %2.2f, bestUAR %2.2f" % (epoch, loss, uars[-1], max(uars)))
            score_file.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                             "—— %d epoch, LR %f, LOSS %f, UAR %2.2f, bestUAR %2.2f\n" % (
                                 epoch, lr, loss, uars[-1], max(uars)))
            score_file.flush()
    score_file.write('\n')
    score_file.close()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='SpearEmotionRecognition')

    # 训练参数设置
    parser.add_argument('--epoch', type=int, default=40, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=0, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')

    # 训练、开发、测试集路径, save_path
    parser.add_argument('--wav_path', type=str, default='dist/adjustwav', help='Path of the WAV file')
    parser.add_argument('--train_path', type=str, default='dist/lab/train.csv', help='Path of the train label file')
    parser.add_argument('--devel_path', type=str, default='dist/lab/devel.csv', help='Path of the devel label file')
    parser.add_argument('--test_path', type=str, default='dist/lab/test.csv', help='Path of the test label file')
    parser.add_argument('--save_path', type=str, default="exps/exp-twochannel", help='Path to save the score.txt and models')

    # 选用已有特征集
    parser.add_argument('--feature_type', type=str, default='xbow', help='选择已有的特征集, 默认None表示不用')
    parser.add_argument('--feature_path', type=str, default='dist/features/', help='特征集位置')

    # 模型参数
    parser.add_argument('--mel', type=str, default='mfcc', help='mfcc or fbank')
    parser.add_argument('--n_mel', type=str, default=40, help='nmel')
    parser.add_argument('--model', type=str, default='twochannel', help='Learning rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

    args = parser.parse_args()
    args = init_args(args)

    st = time.time()
    main(args)
    print(f'模型训练结束，共耗时：{round(time.time() - st, 2)}s')
