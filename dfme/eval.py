import torch
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import argparse
from approximate_gradients import *
from train import test
from matplotlib import pyplot
from dataloader import get_dataloader
from my_utils import *
import os

def myprint(a):
    """Log the print statements"""
    global file
    print(a)

def generate(teacher, student, generator, visualize=False, n_images=10000, test_loader=None, batch_size=250, nz=256, device='cpu', save_dir=None):
    teacher.eval()
    student.eval()
    generator.eval()

    n_iter = n_images // batch_size
    zs = []
    labels = []
    for i in tqdm(range(n_iter), desc='Generation'):
        z = torch.randn(batch_size, nz).to(device)
        label = torch.randint(0, 10, (batch_size, )).to(device)
        logit, x_feat = student(generator(z, label).detach(), is_feat=True)
        # print(logit.shape, x_feat.shape)
        # label = logit.argmax(1)
        if visualize:
            zs.append(x_feat.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    # print(labels)
    # exit(-1)
    x_reals = []
    y_reals = []
    for data in tqdm(test_loader, desc='Real images'):
        data, label = data
        x = data.to(device)
        y = label.to(device)
        # print(x.shape, y.shape)
        # assert len(y.shape) == 1
        logit, x_feat = teacher(x, is_feat=True)
        if visualize:
            x_reals.append(x_feat.detach().cpu().numpy())
            y_reals.append(y.cpu().numpy())

    
    zs = np.vstack(zs)
    labels = np.hstack(labels)
    x_reals = np.vstack(x_reals)
    y_reals = np.hstack(y_reals)
    # print(x_reals.shape, y_reals.shape)
    assert x_reals.shape[0] == y_reals.shape[0]
    if visualize:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tsne = TSNE(n_components=2, n_iter=100000, verbose=1)
        X_res = tsne.fit_transform(zs)
        # X_res = X_res.embedding_
        
        pyplot.scatter(X_res[:, 0], X_res[:, 1], c=labels)
        # pyplot.scatter(Y_ori[np.logical_or(labels_ori == a, labels_ori == b)][:, 0], Y_ori[np.logical_or(labels_ori == a, labels_ori == b)][:, 1], c=labels_ori[np.logical_or(labels_ori == a, labels_ori == b)]+2)
        pyplot.xlabel('Component 1')
        pyplot.ylabel('Component 2')
        pyplot.savefig('{}/tsne_gen.png'.format(save_dir))
        pyplot.close()

        tsne_real = TSNE(n_components=2, n_iter=20000, init='pca', verbose=1)
        X_real = tsne_real.fit_transform(x_reals)
        # X_real = X_real.embedding_
        pyplot.scatter(X_real[:, 0], X_real[:, 1], c=y_reals)
        # pyplot.scatter(Y_ori[np.logical_or(labels_ori == a, labels_ori == b)][:, 0], Y_ori[np.logical_or(labels_ori == a, labels_ori == b)][:, 1], c=labels_ori[np.logical_or(labels_ori == a, labels_ori == b)]+2)
        pyplot.xlabel('Component 1')
        pyplot.ylabel('Component 2')
        pyplot.savefig('{}/tsne_real.png'.format(save_dir))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation process.')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')
    parser.add_argument('--student_model', type=str, default='resnet18_8x', help='Student model architecture (default: resnet18_8x)')
    parser.add_argument('--model', type=str, default='resnet34_8x', choices=classifiers, help='Target model name (default: resnet34_8x)')

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")
    parser.add_argument('--data_root', type=str, default="debug")
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--save_path', type=str, default="/data/lijingru/dfkd/dfme/save_results/tsne_result/")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_images', type=int, default=10000, help="how many images to be generated.")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--vis', action="store_true", help="visualization of tsne results.")
    parser.add_argument('--batch_size', type=int, default=250, help="Batch size for generation.")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.device if use_cuda else "cpu")
    num_classes = 10 if args.dataset in ['cifar10', 'svhn'] else 100
    args.num_classes = num_classes

    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
        if args.dataset == 'svhn':
            print("Loading SVHN TEACHER")
            args.ckpt = 'checkpoint/teacher/svhn-resnet34_8x.pt'
        teacher.load_state_dict( torch.load( args.ckpt, map_location=device) )
    else:
        teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)

    teacher.eval()
    teacher = teacher.to(device)
    args.device = device

    # Eigen values and vectors of the covariance matrix
    _, test_loader = get_dataloader(args)
    myprint("Teacher restored from %s"%(args.ckpt)) 
    print(f"\n\t\tTraining with {args.model} as a Target\n") 
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset),accuracy))
    
    student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes)
    
    # generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32, activation=torch.tanh)
    generator = network.gan.GeneratorC(nz=args.nz, nc=3, img_size=32)

    
    student = student.to(device)
    generator = generator.to(device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    
    if args.student_load_path :
        # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
        student.load_state_dict( torch.load( args.student_load_path , map_location=device))
        myprint("Student initialized from %s"%(args.student_load_path))
        acc = test(args, student=student, generator=generator, device = device, test_loader = test_loader, log=False)

    generate(teacher, student, generator, visualize=True, n_images=args.n_images, test_loader=test_loader, device=device, save_dir=args.save_path, batch_size=args.batch_size)