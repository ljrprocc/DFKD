import os
import shutil

if __name__ == '__main__':
    root = '/data/lijingru/timagenet/tiny-imagenet-200/'
    txt_file = 'val_annotations.txt'
    
    w = os.listdir(os.path.join(root, 'train'))
    for l in w:
        k = os.listdir(os.path.join(root, 'train', l))
        for fs in k:
            if fs.endswith('.txt'):
                shutil.move(os.path.join(root, 'train', l, fs), os.path.join(root, 'train', l, 'images', fs))
            else:
                c = os.path.join(root, 'train', l, fs)
                c = os.listdir(c)
                for fss in c:
                    src = os.path.join(root, 'train', l, fs, fss)
                    dst = os.path.join(root, 'train', l, fss)
                    
                    shutil.move(src, dst)           