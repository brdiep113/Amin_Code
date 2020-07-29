import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pixel unshuffle operator opossite of nn.PixelShuffle
def pixel_unshuffle(fm, r):
    bs, ch, h, w = fm.size()  
    fm_view = fm.contiguous().view(bs, ch, h//r, r, w//r, r)
    fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(bs,
                                                              ch*(r**2),
                                                              h//r,
                                                              w//r)
    return fm_prime


def loss_position(logits, labels):
    '''
    Arguments: 
        - logits: [batch_size, 65, w, h]
        - labels: [batch_size, 1, w*8, h*8]
    Returns:
        - loss: scalar
    '''
    # Convert boolean labels to indices including "no interest point" dustbin
    labels = pixel_unshuffle(labels, 8)
    bs, _, h, w = labels.size()
    labels = torch.cat([2*labels, torch.ones(bs,1,h,w,device=device)], dim=1)
    # argmax semi-randomly returns one of indices if there are more than 1 point
    # in 8x8 negihborhoods
    labels = torch.argmax(labels,dim=1)
    # w = torch.ones(65, device=device) * 3500.0
    # w[64] = 1.0
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss