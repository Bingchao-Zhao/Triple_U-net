import skimage.morphology as sm
import numpy as np 
import cv2
import csv
import os
import torch
import random
import time
SQUARE_KERNEL_KEYWORD = 'square_conv.weight'
last_color=31
def rtime_print(str,end='\r'):
    print('\033[5;{};40m{}\033[0m'.format(random.randint(31, 37),str), end=end,flush=True)
def note_by_split(num,split):
    if num==0:
        return 
    if num%split==0:
        my_print('handling :{}'.format(num))
def get_filename(path,contain_dir=False,abspath=False,num_only=False,no_num=False):
    if not os.path.exists(path):
        my_error('{} not exit!!!'.format(filename))
        return []
    FileNames=os.listdir(path)
    ret=[]
    num=0
    for i in range(len(FileNames)):
        f=os.path.join(os.path.join(path,FileNames[i]))
        if contain_dir==0:
            if os.path.isdir(f):
                continue
        if not abspath:
            f=FileNames[i]
        ret.append(f)
        num+=1
    if num_only:
        return num
    if no_num:
        return ret
    return ret,num
def get_time(complete=False):
    if not complete:
        return time.strftime("%Y-%m-%d",time.localtime(time.time()))
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
def write_csv(content,filename,ifini=0):
    if ifini:
        with open(filename,'w+')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(content)
        my_print('Write success!')
        return
    with open(filename,'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(content)

    my_print('Write success!')

def separate_stain(im):
    # He = np.array([0.6500286,0.704031,0.2860126])
    # DAB = np.array([0.26814753,0.57031375,0.77642715])
    # Res = np.array([0.0,0.0,0.0])
    # Res = np.array([0.7110272,0.42318153,0.5615672])
    # stain_matrix = [(He/np.linalg.norm(He)).tolist(),
    # (DAB/np.linalg.norm(DAB)).tolist(),(Res/np.linalg.norm(Res)).tolist()];
    H= np.array([0.650,0.704,0.286])
    E= np.array([0.072,0.990,0.105])
    R= np.array([0.268,0.570,0.776])
    # H= np.array([0.490157340,0.768970850,0.410401730])
    # E= np.array([0.04615336,0.84206840,0.53739250])
    # R= np.array([0.00000000,0.00000000,0.0000000])
    HDABtoRGB = [(H/np.linalg.norm(H)).tolist(),(E/np.linalg.norm(E)).tolist(),(R/np.linalg.norm(R)).tolist()]
    stain_matrix=HDABtoRGB
    im_inv=np.linalg.inv(stain_matrix)
    im_temp = (-255)*np.log((np.float64(im)+1)/255)/np.log(255)
    image_out = np.reshape(np.dot(np.reshape(im_temp,[-1,3]) ,
                            im_inv),np.shape(im))
    image_out = np.exp((255-image_out)*np.log(255)/255)
    image_out[image_out>255] = 255;
    return np.uint8(image_out)

def com_str(str,rc=True,sep=' ',last=False):
	global last_color
	if rc:
		if last:
			last_color = last_color
		else:
			last_color = random.randint(31, 37);
		return '\033[1;{}m{}{}\033[0m'.format(last_color,str,sep)
	else:
		return '\033[1;36m{}{}\033[0m'.format(str,sep)
        
def my_print(*args,rc=True,sep=' ',if_last=False):
	for i in range(len(args)-1):
		if i==0:
			print(com_str(args[i],rc,'',last=if_last),end='')
			continue
		print(com_str(args[i],rc,sep,last=if_last),end='')
	print(com_str(args[len(args)-1],rc,sep,last=if_last))
    
def my_error(str):
    print('\033[1;31m{}\033[0m'.format(str))
    
def adjustData(img,mask):
    img = img / 255
    mask=(mask > 200)*1
    return (img,mask)
    
def get_attention(img):
    sep=separate_stain(img)
    sep=np.reshape((sep[:,:,0]<230),[np.shape(img)[0],np.shape(img)[1]])
    #remove_small_objects can only handle bool type image.
    sep = sm.remove_small_objects(sep, min_size=100,connectivity=2)
    kernel = sm.disk(1)
    sep = sm.dilation(sep, kernel)
    sep=imfill(sep)
    return sep

def my_load(model,hdf5):
    ret={}
    if isinstance(hdf5,str):
        ud=torch.load(hdf5)
    else:
        ud=hdf5
    for i in model.state_dict().keys():
        for h in ud.keys():
            if i in h or h in i:
                #print(i,h)
                ret[i]=ud[h]
    return ret   
    
def imfill(im_in):
    if im_in.ndim!=2:
        my_error('Only handle Binary but get image dim:{}!'.format(im_in.ndim))
        return im_in
    if np.max(im_in)>1:
        im_th=im_in
    else:
        im_th=im_in*255
    im_th=im_th.astype(np.uint8)
    h, w = im_in.shape[:2]
    temp = np.zeros((h+2, w+2), np.uint8)
    temp[1:h+1,1:w+1]=im_in
    mask=np.zeros((h+4, w+4), np.uint8)
    cv2.floodFill(temp,mask,(0, 0), 255, cv2.FLOODFILL_FIXED_RANGE)
    im_floodfill_inv = ~temp[1:h+1,1:w+1]
    return (im_floodfill_inv>1)*1

def _fuse_kernel(kernel, gamma, std):
    b_gamma = torch.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = b_gamma.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    b_std = torch.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = b_std.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    return kernel * b_gamma / b_std

def _add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                                        square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

def convert_acnet_weights(hdf5, eps=1e-5):
    train_dict = torch.load(hdf5)
    
    deploy_dict = {}
    square_conv_var_names = [name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        square_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_mean')]
        square_std = torch.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_var')] + eps)
        square_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.weight')]
        square_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.bias')]

        ver_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        ver_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_mean')]
        ver_std = torch.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_var')] + eps)
        ver_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.weight')]
        ver_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.bias')]

        hor_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]
        hor_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_mean')]
        hor_std = torch.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_var')] + eps)
        hor_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.weight')]
        hor_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.bias')]

        fused_bias = square_beta + ver_beta + hor_beta - square_mean * square_gamma / square_std \
                     - ver_mean * ver_gamma / ver_std - hor_mean * hor_gamma / hor_std
        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)
        _add_to_square_kernel(fused_kernel, _fuse_kernel(ver_kernel, ver_gamma, ver_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(hor_kernel, hor_gamma, hor_std))

        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k, v in train_dict.items():
        if 'hor_' not in k and 'ver_' not in k and 'square_' not in k:
            deploy_dict[k] = v
    #print(deploy_dict.keys())
    return deploy_dict




def convert_no_norm_acnet_weights(hdf5, eps=1e-5):
    train_dict = torch.load(hdf5)
    
    deploy_dict = {}
    square_conv_var_names = [name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        ver_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        hor_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]
        
        _add_to_square_kernel(square_kernel, ver_kernel)
        _add_to_square_kernel(square_kernel, hor_kernel)

        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = square_kernel

    for k, v in train_dict.items():
        if 'hor_' not in k and 'ver_' not in k and 'square_' not in k:
            deploy_dict[k] = v
    #print(deploy_dict.keys())
    return deploy_dict



