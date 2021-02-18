from dataloader.datasets import CrackSegmentation
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'TunnelCrack':
        train_set = CrackSegmentation(args, split='train')
        val_set = CrackSegmentation(args, split='train')
        test_set = CrackSegmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class