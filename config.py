'''
根据你要训练的车牌类型来设置如下参数
'''
import string

# 图像的通道数，32位图像为4，24位图像为3
CHANNELS = 3
# 图像的高度
# HEIGHT = 50
HEIGHT = 40
# 图像的宽度
# WIDTH = 130
WIDTH = 250
# 验证码的长度
# LEN = 5
LEN = 7
# 验证码中的字符类别

# __letters = string.digits + string.ascii_lowercase
__digit = string.digits
__chars = "ABCDEFGHJKLMNPQRSTUVWXYZ"


__chars_china ='皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学'
# print(len(__chars_china))
# print(__digit)
# print(__chars)
# print(__chars_china)


__letters = __chars_china + __chars+__digit
#print(__letters)
# __chars ='ABCDEFGHJKLMNPQRSTUVWXYZN'
# __chars_china =""
#print(__letters)
CHARS = [c for c in __letters]
# print(CHARS)
print(len(CHARS))


