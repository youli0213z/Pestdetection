import torch
pretrained_weights = torch.load('mask_rcnn_swin_tiny_patch4_window7_1x.pth')
num_class = 21
pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.weight'].resize_(num_class+1,1024)
pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.weight'].resize_(num_class*4,1024)
pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.bias'].resize_(num_class*4)
pretrained_weights['state_dict']['roi_head.mask_head.conv_logits.weight'].resize_(num_class,256,1,1)
pretrained_weights['state_dict']['roi_head.mask_head.conv_logits.bias'].resize_(num_class)
torch.save(pretrained_weights, "mask_rcnn_swin_%d.pth"%num_class)
