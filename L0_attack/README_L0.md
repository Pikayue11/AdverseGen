# L0_attack

### network: resnet18 

### database: cifar-10

A non-target attack 



To test, please run `test_api.py`,   

```
python3 test_api.py
```

it will output 4 numpy arrays, they are

+ the original labels of the input images
+ the new labels of the images after attack
+ number of changed pixels (L0_norm value)
+ if the attack is success ( 1 indicates success while 0 means failure) 

And the according adversarial examples will be saved in the`results` folder





You are recommended to open `test_api.py` to see more details,   

+ you can changes the value of N to change the number of the pictures as input (I provided 1 and 2 pictures), and I put some cifar-10 pictures in `images` folder

+ the most important function is `L0_api()` which I defined in `L0_API.py`, it is read-friendly, you are recommended to read it ( find `L0_api()` first, and then read others in  `L0_API.py` base on it ). 

+ the parameter of  `L0_api()` is a numpy array whose shape is [N, 32, 32, 3], N is the number of  images

  