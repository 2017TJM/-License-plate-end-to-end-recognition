from my_model import MyModel
from torchvision.transforms import functional
import numpy as np
import torch
import string
__digit = string.digits
__chars = "ABCDEFGHJKLMNPQRSTUVWXYZ"

__chars_china ='皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学'
from PIL import Image


class Predictor(object):
    def __init__(self, model_path, gpu=False):
        self.net = MyModel(gpu)
        self.net.load(model_path)
        
    def identify(self, img_path):
        img = Image.open(img_path)

        # to tensor
        np_img = np.asarray(img)
        image = np_img.transpose((2, 0, 1))  # H x W x C  -->  C x H x W
        img = torch.from_numpy(image).float()

        # normalize
        # img = functional.normalize(img, [127.5, 127.5, 127.5, 127.5], [128, 128, 128, 128])
        img = functional.normalize(img, [127.5, 127.5, 127.5], [128, 128, 128])
        if self.net.gpu == True:  # to cpu
            img = img.to('cuda')

        with torch.no_grad():
            xb = img.unsqueeze(0)
            # out = self.net(xb).squeeze(0).view(5, 36)
            out = self.net(xb).squeeze(0).view(7, 67)
            _, predicted = torch.max(out, 1)
            print(predicted)
            print(predicted.tolist())
            # letters = string.digits + string.ascii_lowercase
            letters = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'
            CHARS = [c for c in letters]
            print(CHARS)
            ans = [CHARS[i] for i in predicted.tolist()]
            return ans


if __name__ == '__main__':
    # 设置模型的路径
    model_path = 'chepai_net_31_86.pth'
    # 设置要测试的图像路径
    # img_path = './data/test/3edpg.png'
    #img_path ='./data/voc/VOCdevkit/VOC2019/JPEGImages/0529-7_14-209&440_539&574-539&574_220&533_209&440_528&481-0_0_26_30_26_9_26-127-179.jpg'
    img_path ='./1.jpg'
    man = Predictor(model_path, gpu=False)
    
    print(man.identify(img_path))
        
