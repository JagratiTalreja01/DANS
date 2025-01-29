import torch
import numpy as np
import utility
import data
import model
import loss
import random
from option import args
from trainer import Trainer

#torch.manual_seed(args.seed)
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)
#torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
#np.random.seed(args.seed)  # Numpy module.
#random.seed(args.seed)  # Python random module.
#torch.manual_seed(args.seed)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
