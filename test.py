import numpy as np
import argparse
import os
import cv2
from dataloader.datasets import CrackSegmentation
from torch.utils.data import DataLoader
from model.linkcrack import *

class Predictor(object):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args

        self.model = LinkCrack()

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if args.pretrained_model:
            checkpoint = torch.load(args.pretrained_model)
            self.model.load_state_dict(checkpoint)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}

        test_set = CrackSegmentation(args, split='test')

        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    def val_op(self, input):

        pred = self.model(input)

        pred_mask = pred[0]
        pred_connected = pred[1]

        return torch.cat((pred_mask.clone(), pred_connected.clone()), 1)

    def do(self):
        self.model.eval()

        with torch.no_grad():
            for idx, sample in enumerate(self.test_loader):
                img = sample['image']
                lab = sample['label']

                val_data, val_target = img.type(torch.cuda.FloatTensor).to(self.device), [
                    lab[0].type(torch.cuda.FloatTensor).to(self.device),
                    lab[1].type(torch.cuda.FloatTensor).to(self.device)]
                val_pred = self.val_op(val_data)
                img_cpu = val_data.cpu().squeeze().numpy() * 255
                test_pred = torch.sigmoid(val_pred[:, 0, :, :].cpu().squeeze())
                save_name = os.path.join(self.args.save_path, '%04d.png' % idx)
                test_pred = test_pred.numpy()
                img_cpu = np.transpose(img_cpu, [1, 2, 0])
                img_cpu[test_pred > self.args.acc_sigmoid_th, :] = [255, 0, 0]
                cv2.imwrite(save_name, img_cpu.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description="LinkCrack Training")

    parser.add_argument('--dataset', type=str, default='TunnelCrack',
                        choices=['TunnelCrack'],
                        help='dataset name (default: TunnelCrack)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')

    # cuda, seed and logging
    parser.add_argument('--cuda', action='store_true', default=
    True, help='Use CUDA ')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--acc_sigmoid_th', type=float, default=0.5,
                        help='maximum number of checkpoints to be saved')
    # checking point
    parser.add_argument('--pretrained-model', type=str,
                        default='Tunnel_Crack_FT.pth',
                        help='put the path to resuming file if needed')

    parser.add_argument('--save-path', type=str, default='results',
                        help='put the path to resuming file if needed')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    predict = Predictor(args)
    predict.do()


if __name__ == "__main__":
    main()
