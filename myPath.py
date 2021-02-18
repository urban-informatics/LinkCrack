class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'TunnelCrack':
            return '/home/yyh/TunnelCrack/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError