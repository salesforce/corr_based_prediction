ReadMe for the paper "Entropy Penalty: Towards Generalization Beyond the IID Assumption": 

Version of softwares used:

1. Python 3.6.8
2. PyTorch 1.0.0


Commands:

1. Generate Colored MNIST:
python gen_color_mnist.py

2. Sample command to run Entropy Penalty:
python main.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --save_dir ep --beta 0.1

3. Sample commands to run baseline methods:

- MLE:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --bs 128 --save_dir mle

- AdaBN:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --bs 32 --save_dir adabn --bn --bn_eval

- ALP:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --save_dir alp --alp --nsteps 20 --stepsz 2 --epsilon 8 --beta 0.1

- CLP:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --save_dir clp --clp --beta 0.5

- PGD:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.0001 --save_dir pgd --pgd --nsteps 20 --stepsz 2 --epsilon 8

- VIB:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.001 --save_dir inp --inp_noise 0.2

- Input Noise:

python baseline.py --dataset fgbg_cmnist_cpr0.5-0.5 --seed 0 --root_dir cmnist --lr 0.001 --save_dir inp_noise --inp_noise 0.2


4. Evaluate a trained model on another dataset (here [root_dir] and [save_dir] should be the directoris in which the model to be used is saved):
python eval.py --root_dir cmnist --save_dir ep --dataset mnistm
