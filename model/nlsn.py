from model import common
from model import attention
import torch.nn as nn

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return NLSN(args, dilated.dilated_conv)
    else:
        return NLSN(args)


class NLSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLSN, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=args.chunk_size, n_hashes=args.n_hashes, reduction=4, res_scale=args.res_scale)]         

        for i in range(n_resblock):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
            if (i+1)%8==0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=args.chunk_size, n_hashes=args.n_hashes, reduction=4, res_scale=args.res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x) #[2, 3, 64, 64]
        x = self.head(x)     #[2, 3, 64, 64]

        res = self.body(x)   #[2, 64, 64,64]
        res += x             #[2, 64, 64,64]

        x = self.tail(res)   #[2, 64, 64,64]
        x = self.add_mean(x) #[2, 3, 256, 256]

        return x             #[2, 3, 256, 256]

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

