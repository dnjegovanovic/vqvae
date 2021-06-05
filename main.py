from scipts import  trian
from scipts import eval
import argparse

def run_main():
    trian.train_vqvae()

def eval_main():
    eval.eval()


parser = argparse.ArgumentParser()
parser.add_argument('--train-vqvae', default=0, type=int)
parser.add_argument('--eval-vqvae', default=1, type=int)
parser.add_argument('--generate-new', default=0, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    if args.train_vqvae == 1:
        run_main()
    elif args.eval_vqvae == 1:
        eval_main()
    elif args.generate_new == 1:
        pass
