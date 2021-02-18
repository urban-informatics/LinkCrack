import time
import os
from os import path
import torch
from utils.paths import process
from queue import Queue


class Checkpointer(object):
    r"""Checkpointer for objects using torch serialization.

    Args:
        name (str): Name of the checkpointer. This will also be used for
            the checkpoint filename. # 检查点文件名
        directory (str, optional): Parent directory of where the checkpoints will happen. # 检查点的父目录,默认为当前目录
            A new sub-directory called checkpoints will be created (default '.')
        overwrite (bool, optional): Overwrite/remove the previous checkpoint (default True). # 覆盖/删除上一个检查点
        verbose (bool, optional): Print a statement when loading a checkpoint (default True).  # 加载检查点时打印语句
        timestamp (bool, optional): Add a timestamp to checkpoint filenames (default False).  # 为检查点添加时间戳
        add_count (bool, optional): Add (zero-padded) counter to checkpoint filenames (default True). # 向检查点添加计数
        max_queue (int, optional
        #format [name,_,tag,_,counter, _%Y-%m-%d-%H-%M-%S]

    Example:
        >>> a = {'data': 5}
        >>> a_chkp = dlt.util.Checkpointer('a_saved')
        >>> a_chkp.save(a)
        >>> b = a_chkp.load()
        {'data': 5}

    It automatically saves and loads the `state_dict` of objects::

        >>> net = nn.Sequential(nn.Linear(1,1), nn.Sigmoid())
        >>> net_chkp = dlt.util.Checkpointer('net_state_dict')
        >>> net_chkp.save(net)
        >>> net_chkp.load(net)
        Sequential(
            (0): Linear(in_features=1, out_features=1)
            (1): Sigmoid()
        )
        >>> state_dict = net_chkp.load()
        OrderedDict([('0.weight',
        -0.2286
        [torch.FloatTensor of size 1x1]
        ), ('0.bias',
        0.8495
        [torch.FloatTensor of size 1]
        )])

    Warning:
        If a model is wrapped in `nn.DataParallel` then the wrapped model.module
        (state_dict) is saved. Thus applying `nn.DataParallel` must be done after
        using `Checkpointer.load()`.

    """

    def __init__(self, name, directory='.', overwrite=False, verbose=True, timestamp=False, add_count=True,
                 max_queue=None):
        self.name = name
        self.directory = process(directory, create=True)
        self.directory = path.join(self.directory, 'checkpoints')
        self.directory = process(self.directory, create=True)
        self.overwrite = overwrite
        self.timestamp = timestamp
        self.add_count = add_count
        self.verbose = verbose
        self.chkp = path.join(self.directory, ".{0}.chkp".format(self.name))
        self.counter, self.filename = torch.load(self.chkp) if path.exists(self.chkp) else (0, '')
        self.save_queue = False

        if overwrite == False and isinstance(max_queue, int) and max_queue > 1:
            self.save_queue = True
            self._queue = Queue(max_queue)
        elif max_queue is not None:
            print('WARNING: illegal max_queue Value!.')

        self.show_save_pth_name = ''

    def _say(self, line):
        if self.verbose:
            print(line)

    def _make_name(self, tag=None):
        strend = ''
        if tag is not None:
            strend += '_' + str(tag)
        if self.add_count:
            strend += "_{:07d}".format(self.counter)
        if self.timestamp:
            strend += time.strftime("_%Y-%m-%d-%H-%M-%S")
        filename = "{0}{1}.pth".format(self.name, strend)
        self.filename = path.join(self.directory, filename)
        return self.filename

    def _set_state(self, obj, state):
        if hasattr(obj, 'load_state_dict'):
            obj.load_state_dict(state)
            return obj
        else:
            return state

    def _get_state(self, obj):
        if isinstance(obj, torch.nn.DataParallel):
            obj = obj.module
        if hasattr(obj, 'state_dict'):
            return obj.state_dict()
        else:
            return obj

    def __call__(self, obj, tag=None, *args, **kwargs):
        """Same as :meth:`save`"""
        self.save(obj, tag=tag, *args, **kwargs)

    def save(self, obj, tag=None, *args, **kwargs):
        """Saves a checkpoint of an object.

        Args:
            obj: Object to save (must be serializable by torch).
            tag (str, optional): Tag to add to saved filename (default None).
            args: Arguments to pass to `torch.save`.
            kwargs: Keyword arguments to pass to `torch.save`.

        """

        self.counter += 1
        old_filename = self.filename
        new_filename = self._make_name(tag)
        if self.save_queue is True:
            if self._queue.full() is True:
                delete_name = self._queue.get()
                try:
                    os.remove(delete_name)
                except:
                    pass
            self._queue.put(new_filename)

        if new_filename == old_filename and not self.overwrite:
            print('WARNING: Overwriting file in non overwrite mode.')
        elif self.overwrite:
            try:
                os.remove(old_filename)
            except:
                pass

        torch.save(self._get_state(obj), new_filename, *args, **kwargs)
        torch.save((self.counter, new_filename), self.chkp)

        self.show_save_pth_name = new_filename

    def load(self, obj=None, preprocess=None, multi_gpu=False, *args, **kwargs):
        """Loads a checkpoint from disk.

        Args:
            obj (optional): Needed if we load the `state_dict` of an `nn.Module`.
            preprocess (optional): Callable to preprocess the loaded object.
            args: Arguments to pass to `torch.load`.
            kwargs: Keyword arguments to pass to `torch.load`.

        Returns:
            The loaded file.

        """
        if isinstance(obj, str) and obj.split('.')[-1] == 'pth':  # 正常加载
            self._say("Loaded checkpoint: {0}".format(obj))
            obj = torch.load(obj)
        elif self.counter > 0 and obj is None:
            loaded = torch.load(self.filename, *args, **kwargs)
            if preprocess is not None:
                loaded = preprocess(loaded)
            obj = self._set_state(obj, loaded)
            self._say("Loaded {0} checkpoint: {1}".format(self.name, self.filename))

        if multi_gpu is True:
            from collections import OrderedDict
            multi_gpu_obj = OrderedDict()
            for k, v in obj.items():
                name = 'module.' + k  # add `module.`
                multi_gpu_obj[name] = v
            obj = multi_gpu_obj


        return obj


if __name__ == '__main__':

    import torch.nn as nn

    net = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
    m_net = nn.DataParallel(net)
    # s = Checkpointer('save')
    # s.save(m_net)
    # a = s.load() # 不带module,module

    ### 正确用法

    ss = Checkpointer(name='SegNet', directory='./', overwrite=False, verbose=True, timestamp=True, max_queue=5)

    for step in range(100):
        if step % 10 == 0 and step != 0:
            ss.save(m_net, tag='%d' % step)
    ss.load('./checkpoints/SegNet_10_0000028_2019-06-27-01-44-55.pth')
